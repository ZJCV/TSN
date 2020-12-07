# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午4:03
@file: video_visualizor.py
@author: zj
@description: 
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from .img_visualizer import ImgVisualizer

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (0, 0, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


class VideoVisualizer:

    def __init__(self, cfg, colormap="rainbow", top_k=5):
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        self.class_names = self.get_class_names(cfg.VISUALIZATION.LABEL_FILE_PATH)
        self.color_map = plt.get_cmap(colormap)
        self.top_k = top_k

    def get_class_names(self, path):
        assert os.path.exists(path), f'{path} is None'
        with open(path, 'r') as f:
            class_names = [line.strip().split(' ')[1] for line in f]

        return class_names

    def __call__(self, task):
        frames = task.frames
        preds = task.action_preds

        buffer = frames[: task.num_buffer_frames]
        frames = frames[task.num_buffer_frames:]

        frames = self.draw_clip(frames, preds)
        task.frames = np.array(buffer + frames)

        return task

    def draw_clip(self, frames, preds):
        """
            Draw predicted labels to clip.
            Args:
                frames (array-like): video data in the shape (T, H, W, C).
                preds (tensor): For recognition task, input shape can be (num_classes,).
        """
        top_scores, top_classes = torch.topk(preds, k=self.top_k)
        top_scores, top_classes = top_scores.tolist(), top_classes.tolist()
        # Create labels top k predicted classes with their scores.

        texts = self.create_text_labels(
            top_classes[0],
            top_scores[0],
            self.class_names
        )
        pred_class = top_classes[0]
        colors = [self._get_color(pred) for pred in pred_class]
        font_size = min(
            max(np.sqrt(frames[0].shape[0] * frames[0].shape[1]) // 35, 5), 9
        )

        print(texts, pred_class)
        img_ls = list()
        for frame in frames:
            for i, (text, color) in enumerate(zip(texts, colors)):
                location = (0, 100 + i * 20)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
            img_ls.append(frame)

        # img_ls = []
        # for frame in frames:
        #     start = time.time()
        #     draw_img = self.draw_one_frame(frame, preds)
        #     end = time.time()
        #     print('darw one frame need: {}'.format(end - start))
        #
        #     img_ls.append(draw_img)

        return img_ls

    def draw_one_frame(self, frame, preds, text_alpha=0.7):
        """
            Draw labels for one image. By default, predicted labels are drawn in
            the top left corner of the image
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
                text_alpha (Optional[float]): transparency level of the box wrapped around text labels.
        """
        top_scores, top_classes = torch.topk(preds, k=self.top_k)
        # print('preds: ', preds)
        # print('top_scores: ', top_scores)
        # print('top_classes: ', top_scores)
        top_scores, top_classes = top_scores.tolist(), top_classes.tolist()
        # Create labels top k predicted classes with their scores.
        text_labels = []
        text_labels.append(
            self.create_text_labels(
                top_classes[0],
                top_scores[0],
                self.class_names
            )
        )
        frame_visualizer = ImgVisualizer(frame, meta=None)
        font_size = min(
            max(np.sqrt(frame.shape[0] * frame.shape[1]) // 35, 5), 9
        )
        top_corner = False

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

    def create_text_labels(self, classes, scores, class_names, ground_truth=False):
        """
        Create text labels.
        Args:
            classes (list[int]): a list of class ids for each example.
            scores (list[float] or None): list of scores for each example.
            class_names (list[str]): a list of class names, ordered by their ids.
            ground_truth (bool): whether the labels are ground truth.
        Returns:
            labels (list[str]): formatted text labels.
        """
        try:
            labels = [class_names[i] for i in classes]
        except IndexError:
            print("Class indices get out of range: {}".format(classes))
            return None

        if ground_truth:
            labels = ["[{}] {}".format("GT", label) for label in labels]
        elif scores is not None:
            assert len(classes) == len(scores)
            labels = [
                "[{:.2f}] {}".format(s, label) for s, label in zip(scores, labels)
            ]
        return labels

    def _get_color(self, class_id):
        """
        Get color for a class id.
        Args:
            class_id (int): class id.
        """
        return self.color_map(class_id / self.num_classes)[:3]
