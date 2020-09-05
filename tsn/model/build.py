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


def build_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES
    backbone = cfg.MODEL.BACKBONE
    consensus = cfg.MODEL.CONSENSUS

    return TSN(num_classes=num_classes, backbone=backbone, consensus=consensus)


def build_criterion(cfg):
    return nn.CrossEntropyLoss()
