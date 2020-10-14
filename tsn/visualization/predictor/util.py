#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import numpy as np

import tsn.util.logging as logging

logger = logging.get_logger(__name__)


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
