# -*- coding: utf-8 -*-

"""
@date: 2020/10/14 下午4:45
@file: util.py
@author: zj
@description: 
"""

import os
import json
import torch

import tsn.util.logging as logging

logger = logging.get_logger(__name__)


def get_class_names(path):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
    Returns:
        class_names (list of strs): list of class names.
    """
    assert os.path.exists(path), f'{path} is None'
    try:
        with open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    class_names = [None] * len(class2idx)

    for k, i in class2idx.items():
        class_names[i - 1] = k

    return class_names


def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the manager visualizer object.
    """
    frames = task.frames
    preds = task.action_preds

    for i in range(len(preds)):
        preds[i] = torch.softmax(preds[i], dim=0)

    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames:]

    frames = video_vis.draw_clip(frames, preds, text_alpha=1)
    del task

    return buffer + frames


def create_text_labels(classes, scores, class_names):
    """
    Create text labels.
    Args:
        classes (list[int]): a list of class ids for each example.
        scores (list[float] or None): list of scores for each example.
        class_names (list[str]): a list of class names, ordered by their ids.
    Returns:
        labels (list[str]): formatted text labels.
    """
    try:
        labels = [class_names[i] for i in classes]
    except IndexError:
        logger.error("Class indices get out of range: {}".format(classes))
        return None

    if scores is not None:
        assert len(classes) == len(scores)
        labels = [
            "[{:.2f}] {}".format(s, label) for s, label in zip(scores, labels)
        ]
    return labels
