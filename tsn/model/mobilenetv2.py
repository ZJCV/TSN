# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午6:41
@file: mobilenetv2.py
@author: zj
@description: 
"""

import torch
import torchvision.models as models

from .consensus import Consensus


class MobileNetV2(models.MobileNetV2):

    def __init__(self, num_classes=1000, consensus='avg'):
        super(MobileNetV2, self).__init__(num_classes=num_classes)

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
            prob_list.append(self._forward_impl(data))

        probs = self.consensus(torch.stack(prob_list))
        return probs
