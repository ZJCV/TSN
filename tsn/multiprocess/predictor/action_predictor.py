# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午3:59
@file: action_predictor.py
@author: zj
@description: 
"""

import torch
import cv2
import numpy as np
from operator import itemgetter

from tsn.model.recognizers.build import build_recognizer
from tsn.util.distributed import get_device, get_local_rank
from tsn.data.transforms.build import build_transform


class ActionPredictor:

    def __init__(self, cfg):
        device = get_device((get_local_rank()))

        self.model = build_recognizer(cfg, device)
        self.model.eval()
        self.transform = build_transform(cfg, is_train=False)

        with open(cfg.VISUALIZATION.LABEL_FILE_PATH, 'r') as f:
            self.label = [line.strip().split(' ')[1] for line in f]

        self.cfg = cfg
        self.device = device
        self.cpu_device = torch.device('cpu')

    @torch.no_grad()
    def __call__(self, task):
        frames = task.frames

        inputs = self.process_cv2_inputs(frames).to(device=self.device, non_blocking=True)
        preds = self.model(inputs)['probs']
        preds = torch.softmax(preds, dim=1).to(self.cpu_device)

        # num_selected_labels = min(len(self.label), 5)
        # scores_tuples = tuple(zip(self.label, preds))
        # scores_sorted = sorted(
        #     scores_tuples, key=itemgetter(1), reverse=True)
        # results = scores_sorted[:num_selected_labels]

        task.add_action_preds(preds)

        return task

    def process_cv2_inputs(self, frames):
        num_clips = self.cfg.DATASETS.NUM_CLIPS
        input_format = self.cfg.VISUALIZATION.INPUT_FORMAT

        index = np.linspace(0, len(frames) - 1, num=num_clips).astype(np.int)
        if input_format == "BGR":
            frames = [
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in index
            ]

        image_list = [self.transform(frame) for frame in frames]
        # [T, C, H, W] -> [C, T, H, W]
        image = torch.stack(image_list).transpose(0, 1)
        # [C, T, H, W] -> [1, C, T, H, W]
        inputs = image.unsqueeze(0)
        return inputs
