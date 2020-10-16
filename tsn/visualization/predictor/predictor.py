#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2
import torch

from tsn.model.build import build_model
from tsn.util import logging
from tsn.visualization.predictor.util import process_cv2_inputs

logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the manager model and print model statistics.
        self.model = build_model(cfg, gpu_id)
        self.model.eval()
        self.cfg = cfg

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
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

        inputs = process_cv2_inputs(frames, self.cfg)

        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.to(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        with torch.no_grad():
            preds = self.model(inputs)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()

        preds = preds.detach()
        task.add_action_preds(preds)

        return task
