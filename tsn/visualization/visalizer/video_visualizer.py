#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import torch

import tsn.util.logging as logging
from .util import get_class_names, create_text_labels
from .img_visualizer import ImgVisualizer

logger = logging.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)


class VideoVisualizer:
    def __init__(
            self,
            num_classes,
            class_names_path,
            colormap="rainbow",
            thres=0.7,
            lower_thres=0.3,
            common_class_names=None
    ):
        """
        Args:
            num_classes (int): total number of classes.
            class_names_path (str): path to json file that maps class names to ids.
                Must be in the format {classname: id}.
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

        """
        self.num_classes = num_classes
        self.class_names = get_class_names(class_names_path)
        self.thres = thres
        self.lower_thres = lower_thres

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
            text_alpha=0.7,
    ):
        """
            Draw labels for one image. By default, predicted labels are drawn in
            the top left corner of the image.
            Args:
                frame (array-like): a tensor or numpy array of shape (H, W, C), where H and W correspond to
                    the height and width of the image respectively. C is the number of
                    color channels. The image is required to be in RGB format since that
                    is a requirement of the Matplotlib library. The image is also expected
                    to be in the range [0, 255].
                preds (tensor or list):
                    For recognition task, input shape can be (num_classes,).
                text_alpha (Optional[float]): transparency level of the box wrapped around text labels.
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

        top_scores, top_classes = [], []
        for pred in preds:
            mask = pred >= self.thres
            top_scores.append(pred[mask].tolist())
            top_class = torch.where(mask)[0].tolist()
            top_classes.append(top_class)

        # Create labels top k predicted classes with their scores.
        text_labels = []
        for i in range(n_instances):
            text_labels.append(
                create_text_labels(
                    top_classes[i],
                    top_scores[i],
                    self.class_names
                )
            )
        frame_visualizer = ImgVisualizer(frame, meta=None)
        font_size = min(
            max(np.sqrt(frame.shape[0] * frame.shape[1]) // 35, 5), 9
        )
        top_corner = True
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

    def draw_clip(
            self,
            frames,
            preds,
            text_alpha=0.5,
            repeat_frame=1,
    ):
        """
            Draw predicted labels to clip.
            Args:
                frames (array-like): manager data in the shape (T, H, W, C).
                preds (tensor): For recognition task, input shape can be (num_classes,).
                text_alpha (float): transparency label of the box wrapped around text labels.
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

        text_alpha = text_alpha
        frames = frames[repeated_seq]
        img_ls = []
        for frame in frames:
            draw_img = self.draw_one_frame(
                frame,
                preds,
                text_alpha=text_alpha
            )
            if adjusted:
                draw_img = draw_img.astype("float32") / 255

            img_ls.append(draw_img)

        return img_ls

    def _adjust_frames_type(self, frames):
        """
            Modify manager data to have dtype of uint8 and values range in [0, 255].
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