# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 下午5:18
@file: video_visualizer.py
@author: zj
@description: 
"""

import itertools
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import torch

from ..utils.misc import _create_text_labels
from .img_visualizer import ImgVisualizer
from ..utils.misc import get_class_names

from tsn.util import logging

logger = logging.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)


class VideoVisualizer:
    def __init__(
            self,
            num_classes,
            class_names_path,
            top_k=1,
            colormap="rainbow",
            thres=0.7,
            lower_thres=0.3,
            common_class_names=None,
            mode="top-k",
    ):
        """
        Args:
            num_classes (int): total number of classes.
            class_names_path (str): path to json file that maps class names to ids.
                Must be in the format {classname: id}.
            top_k (int): number of top predicted classes to plot.
            colormap (str): the colormap to choose color for class labels from.
                See https://matplotlib.org/tutorials/colors/colormaps.html
            thres (float): threshold for picking predicted classes to visualize.
            lower_thres (Optional[float]): If `common_class_names` if given,
                this `lower_thres` will be applied to uncommon classes and
                `thres` will be applied to classes in `common_class_names`.
            common_class_names (Optional[list of str(s)]): list of common class names
                to apply `thres`. Class names not included in `common_class_names` will
                have `lower_thres` as a threshold. If None, all classes will have `thres` as a threshold.
                This is helpful for model trained on highly imbalanced dataset.
            mode (str): Supported modes are {"top-k", "thres"}.
                This is used for choosing predictions for visualization.

        """
        assert mode in ["top-k", "thres"], "Mode {} is not supported.".format(
            mode
        )
        self.mode = mode
        self.num_classes = num_classes
        self.class_names, _, _ = get_class_names(class_names_path, None, None)
        self.top_k = top_k
        self.thres = thres
        self.lower_thres = lower_thres

        if mode == "thres":
            self._get_thres_array(common_class_names=common_class_names)

        self.color_map = plt.get_cmap(colormap)

    def _get_color(self, class_id):
        """
        Get color for a class id.
        Args:
            class_id (int): class id.
        """
        return self.color_map(class_id / self.num_classes)[:3]

    def draw_one_frame(
            self,
            frame,
            preds,
            bboxes=None,
            alpha=0.5,
            text_alpha=0.7,
            ground_truth=False,
    ):
        """
        Draw labels and bouding boxes for one image. By default, predicted labels are drawn in
        the top left corner of the image or corresponding bounding boxes. For ground truth labels
        (setting True for ground_truth flag), labels will be drawn in the bottom left corner.
        Args:
            frame (array-like): a tensor or numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            preds (tensor or list): If ground_truth is False, provide a float tensor of shape (num_boxes, num_classes)
                that contains all of the confidence scores of the model.
                For recognition task, input shape can be (num_classes,). To plot true label (ground_truth is True),
                preds is a list contains int32 of the shape (num_boxes, true_class_ids) or (true_class_ids,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates of the bounding boxes.
            alpha (Optional[float]): transparency level of the bounding boxes.
            text_alpha (Optional[float]): transparency level of the box wrapped around text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
        """
        if isinstance(preds, torch.Tensor):
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            n_instances = preds.shape[0]
        elif isinstance(preds, list):
            n_instances = len(preds)
        else:
            logger.error("Unsupported type of prediction input.")
            return

        if ground_truth:
            top_scores, top_classes = [None] * n_instances, preds

        elif self.mode == "top-k":
            top_scores, top_classes = torch.topk(preds, k=self.top_k)
            top_scores, top_classes = top_scores.tolist(), top_classes.tolist()
        elif self.mode == "thres":
            top_scores, top_classes = [], []
            for pred in preds:
                mask = pred >= self.thres
                top_scores.append(pred[mask].tolist())
                top_class = torch.squeeze(torch.nonzero(mask), dim=-1).tolist()
                top_classes.append(top_class)

        # Create labels top k predicted classes with their scores.
        text_labels = []
        for i in range(n_instances):
            text_labels.append(
                _create_text_labels(
                    top_classes[i],
                    top_scores[i],
                    self.class_names,
                    ground_truth=ground_truth,
                )
            )
        frame_visualizer = ImgVisualizer(frame, meta=None)
        font_size = min(
            max(np.sqrt(frame.shape[0] * frame.shape[1]) // 35, 5), 9
        )
        top_corner = not ground_truth
        if bboxes is not None:
            assert len(preds) == len(
                bboxes
            ), "Encounter {} predictions and {} bounding boxes".format(
                len(preds), len(bboxes)
            )
            for i, box in enumerate(bboxes):
                text = text_labels[i]
                pred_class = top_classes[i]
                colors = [self._get_color(pred) for pred in pred_class]

                box_color = "r" if ground_truth else "g"
                line_style = "--" if ground_truth else "-."
                frame_visualizer.draw_box(
                    box,
                    alpha=alpha,
                    edge_color=box_color,
                    line_style=line_style,
                )
                frame_visualizer.draw_multiple_text(
                    text,
                    box,
                    top_corner=top_corner,
                    font_size=font_size,
                    box_facecolors=colors,
                    alpha=text_alpha,
                )
        else:
            text = text_labels[0]
            pred_class = top_classes[0]
            colors = [self._get_color(pred) for pred in pred_class]
            frame_visualizer.draw_multiple_text(
                text,
                torch.Tensor([0, 5, frame.shape[1], frame.shape[0] - 5]),
                top_corner=top_corner,
                font_size=font_size,
                box_facecolors=colors,
                alpha=text_alpha,
            )

        return frame_visualizer.output.get_image()

    def draw_clip_range(
            self,
            frames,
            preds,
            bboxes=None,
            text_alpha=0.5,
            ground_truth=False,
            keyframe_idx=None,
            draw_range=None,
            repeat_frame=1,
    ):
        """
        Draw predicted labels or ground truth classes to clip. Draw bouding boxes to clip
        if bboxes is provided. Boxes will gradually fade in and out the clip, centered around
        the clip's central frame, within the provided `draw_range`.
        Args:
            frames (array-like): video data in the shape (T, H, W, C).
            preds (tensor): a tensor of shape (num_boxes, num_classes) that contains all of the confidence scores
                of the model. For recognition task or for ground_truth labels, input shape can be (num_classes,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates of the bounding boxes.
            text_alpha (float): transparency label of the box wrapped around text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
            keyframe_idx (int): the index of keyframe in the clip.
            draw_range (Optional[list[ints]): only draw frames in range [start_idx, end_idx] inclusively in the clip.
                If None, draw on the entire clip.
            repeat_frame (int): repeat each frame in draw_range for `repeat_frame` time for slow-motion effect.
        """
        if draw_range is None:
            draw_range = [0, len(frames) - 1]
        if draw_range is not None:
            draw_range[0] = max(0, draw_range[0])
            left_frames = frames[: draw_range[0]]
            right_frames = frames[draw_range[1] + 1:]

        draw_frames = frames[draw_range[0]: draw_range[1] + 1]
        if keyframe_idx is None:
            keyframe_idx = len(frames) // 2

        img_ls = (
                list(left_frames)
                + self.draw_clip(
            draw_frames,
            preds,
            bboxes=bboxes,
            text_alpha=text_alpha,
            ground_truth=ground_truth,
            keyframe_idx=keyframe_idx - draw_range[0],
            repeat_frame=repeat_frame,
        )
                + list(right_frames)
        )

        return img_ls

    def draw_clip(
            self,
            frames,
            preds,
            bboxes=None,
            text_alpha=0.5,
            ground_truth=False,
            keyframe_idx=None,
            repeat_frame=1,
    ):
        """
        Draw predicted labels or ground truth classes to clip. Draw bouding boxes to clip
        if bboxes is provided. Boxes will gradually fade in and out the clip, centered around
        the clip's central frame.
        Args:
            frames (array-like): video data in the shape (T, H, W, C).
            preds (tensor): a tensor of shape (num_boxes, num_classes) that contains all of the confidence scores
                of the model. For recognition task or for ground_truth labels, input shape can be (num_classes,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates of the bounding boxes.
            text_alpha (float): transparency label of the box wrapped around text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
            keyframe_idx (int): the index of keyframe in the clip.
            repeat_frame (int): repeat each frame in draw_range for `repeat_frame` time for slow-motion effect.
        """
        assert repeat_frame >= 1, "`repeat_frame` must be a positive integer."

        repeated_seq = range(0, len(frames))
        repeated_seq = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, repeat_frame) for x in repeated_seq
            )
        )

        frames, adjusted = self._adjust_frames_type(frames)
        if keyframe_idx is None:
            half_left = len(repeated_seq) // 2
            half_right = (len(repeated_seq) + 1) // 2
        else:
            mid = int((keyframe_idx / len(frames)) * len(repeated_seq))
            half_left = mid
            half_right = len(repeated_seq) - mid

        alpha_ls = np.concatenate(
            [
                np.linspace(0, 1, num=half_left),
                np.linspace(1, 0, num=half_right),
            ]
        )
        text_alpha = text_alpha
        frames = frames[repeated_seq]
        img_ls = []
        for alpha, frame in zip(alpha_ls, frames):
            draw_img = self.draw_one_frame(
                frame,
                preds,
                bboxes,
                alpha=alpha,
                text_alpha=text_alpha,
                ground_truth=ground_truth,
            )
            if adjusted:
                draw_img = draw_img.astype("float32") / 255

            img_ls.append(draw_img)

        return img_ls

    def _adjust_frames_type(self, frames):
        """
        Modify video data to have dtype of uint8 and values range in [0, 255].
        Args:
            frames (array-like): 4D array of shape (T, H, W, C).
        Returns:
            frames (list of frames): list of frames in range [0, 1].
            adjusted (bool): whether the original frames need adjusted.
        """
        assert (
                frames is not None and len(frames) != 0
        ), "Frames does not contain any values"
        frames = np.array(frames)
        assert np.array(frames).ndim == 4, "Frames must have 4 dimensions"
        adjusted = False
        if frames.dtype in [np.float32, np.float64]:
            frames *= 255
            frames = frames.astype(np.uint8)
            adjusted = True

        return frames, adjusted

    def _get_thres_array(self, common_class_names=None):
        """
        Compute a thresholds array for all classes based on `self.thes` and `self.lower_thres`.
        Args:
            common_class_names (Optional[list of strs]): a list of common class names.
        """
        common_class_ids = []
        if common_class_names is not None:
            common_classes = set(common_class_names)

            for i, name in enumerate(self.class_names):
                if name in common_classes:
                    common_class_ids.append(i)
        else:
            common_class_ids = list(range(self.num_classes))

        thres_array = np.full(
            shape=(self.num_classes,), fill_value=self.lower_thres
        )
        thres_array[common_class_ids] = self.thres
        self.thres = torch.from_numpy(thres_array)
