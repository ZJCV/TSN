# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2
import torch
import numpy as np

from tsn.data.transforms.build import build_transform
from tsn.model.recognizers.build import build_recognizer
from tsn.util.distributed import get_device, get_local_rank

from tsn.util import logging

logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py.
        """
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if gpu_id is None:
            self.device = get_device(get_local_rank())
        else:
            self.device = torch.device(f"cuda:{gpu_id}")

        # Build the video model and print model statistics.
        self.model = build_recognizer(cfg, device=self.device)
        self.model.eval()
        self.cfg = cfg

        self.transform = build_transform(cfg, is_train=False)

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        frames = task.frames

        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        frames = [
            self.transform(frame)
            for frame in frames
        ]

        # list -> torch(T, C, H, W) -> torch(C, T, H, W) -> torch(1, C, T, H, W)
        inputs = torch.stack(frames).transpose(0, 1).unsqueeze(0).to(device=self.device, non_blocking=True)
        preds = self.model(inputs)['probs'].cpu().detach()

        task.add_action_preds(preds)
        return task
