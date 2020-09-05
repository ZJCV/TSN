# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午9:54
@file: tsn.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from .consensus import Consensus


class TSN(nn.Module):

    def __init__(self, num_classes=1000, backbone='resnet50', consensus='avg'):
        super(TSN, self).__init__()

        self.backbone = self.build_backbone(backbone, num_classes=num_classes)
        self.consensus = Consensus(type=consensus)

    def forward(self, x):
        """
        输入数据大小为NxTxCxHxW，按T维度分别计算NxCxHxW，然后按照融合策略计算最终分类概率
        """
        assert len(x.shape) == 5
        N, T, C, H, W = x.shape[:5]

        input_data = x.transpose(0, 1)
        prob_list = list()
        for data in input_data:
            prob_list.append(self.backbone(data))

        probs = self.consensus(torch.stack(prob_list))
        return probs

    def build_backbone(self, name, num_classes=1000):
        if 'resnet50'.__eq__(name):
            return resnet50(num_classes=num_classes)
