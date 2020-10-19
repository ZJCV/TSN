# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午2:38
@file: crossentropy_loss.py
@author: zj
@description: 
"""

import torch.nn as nn
from tsn.model import registry


@registry.CRITERION.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):

    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, inputs, targets):
        return self.loss(inputs, targets)
