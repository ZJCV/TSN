#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import torch
import numpy as np

from fvcore.common.file_io import PathManager
import tsn.util.logging as logging

logger = logging.get_logger(__name__)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    assert os.path.exists(path), f'{path} is None'
    try:
        with PathManager.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * max_key
    # class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i - 1] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with PathManager.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with PathManager.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def process_cv2_inputs(frames, cfg):
    """
    Normalize and prepare inputs as a list of tensors. Each tensor
    correspond to a unique pathway.
    Args:
        frames (list of array): list of input images (correspond to one clip) in range [0, 255].
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    from tsn.data.transforms.build import build_transform
    transform = build_transform(cfg, is_train=False)

    num_clips = cfg.DATASETS.NUM_CLIPS
    index = np.linspace(0, len(frames) - 1, num=num_clips).astype(np.int)
    image_list = [transform(frames[i]) for i in index]
    # [T, C, H, W] -> [C, T, H, W]
    image = torch.stack(image_list).transpose(0, 1)
    inputs = image.unsqueeze(0)
    return inputs


def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the video visualizer object.
    """
    frames = task.frames
    preds = task.action_preds

    for i in range(len(preds)):
        preds[i] = torch.softmax(preds[i], dim=0)

    draw_range = [
        task.num_buffer_frames,
        len(frames)
    ]
    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames:]

    # frames = video_vis.draw_clip_range(
    #     frames, preds, draw_range=draw_range, text_alpha=1
    # )
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
