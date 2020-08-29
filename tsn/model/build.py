# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torch.nn.modules import Sequential

from .tsn import TSN


def build_model(num_classes=1000):
    return TSN(num_classes=num_classes, backbone='resnet', consensus='avg')


def build_criterion():
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    model = build_model(num_classes=10)
    print(model)

    data = torch.randn((8, 3, 224, 224))
    outputs = model(data)
    print(outputs.shape)
