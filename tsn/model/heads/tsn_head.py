# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午7:38
@file: tsn_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry


@registry.HEAD.register('TSNHead')
class TSNHead(nn.Module):

    def __init__(self, cfg):
        super(TSNHead, self).__init__()

        in_channels = cfg.MODEL.FEATURE_DIMS
        num_classes = cfg.MODEL.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x