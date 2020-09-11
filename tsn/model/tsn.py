# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午9:54
@file: tsn.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from .recognizers.build import build_recognizer
from .consensus.build import build_consensus


class TSN(nn.Module):

    def __init__(self, cfg):
        super(TSN, self).__init__()

        self.num_segs = cfg.DATASETS.NUM_SEGS
        self.modality = cfg.DATASETS.MODALITY

        self.recognizer = build_recognizer(cfg)
        self.consensus = build_consensus(cfg)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        N, T, C, H, W = imgs.shape[:5]

        prob_list = list()
        for i in range(len(self.modality)):
            input_data = imgs[:, i * self.num_segs:(i + 1) * self.num_segs, :, :, :].reshape(-1, C, H, W)
            probs = self.recognizer(input_data).reshape(N, self.num_segs, -1)
            prob_list.append(self.consensus(probs, dim=1))

        probs = self.consensus(torch.stack(prob_list), dim=0)
        return probs
