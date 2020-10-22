#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from tsn.model.build import build_model
from tsn.data.transforms.build import build_transform
from .util import process_cv2_inputs


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                tsn/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()
        self.transform = build_transform(cfg, is_train=False)

        self.cfg = cfg

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor)
        """
        frames = task.frames

        inputs = process_cv2_inputs(frames, self.cfg, self.transform)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        preds = self.model(inputs)
        preds = torch.softmax(preds, dim=1)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()

        preds = preds.detach()
        task.add_action_preds(preds)

        return task
