# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午7:43
@file: tsn_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.nn.modules.module import T

from .. import registry
from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..consensus.build import build_consensus
from ..norm_helper import freezing_bn


@registry.RECOGNIZER.register('TSNRecognizer')
class TSNRecognizer(nn.Module):

    def __init__(self, cfg):
        super(TSNRecognizer, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)
        self.consensus = build_consensus(cfg)

        self.fix_bn = cfg.MODEL.NORM.FIX_BN
        self.partial_bn = cfg.MODEL.NORM.PARTIAL_BN

    def train(self, mode: bool = True) -> T:
        super(TSNRecognizer, self).train(mode=mode)

        if mode and (self.partial_bn or self.fix_bn):
            freezing_bn(self, partial_bn=self.partial_bn)

        return self

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        imgs = imgs.transpose(1, 2)
        N, T, C, H, W = imgs.shape[:5]

        input_data = imgs.reshape(-1, C, H, W)
        features = self.backbone(input_data)
        probs = self.head(features).reshape(N, T, -1)

        probs = self.consensus(probs, dim=1)
        return {'probs': probs}
