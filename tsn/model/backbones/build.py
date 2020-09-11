# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from .resnet import resnet50
from tsn.model import registry


def build_backbone(cfg):
    return registry.BACKBONE[cfg.MODEL.BACKBONE](pretrained=cfg.MODEL.PRETRAINED, partial_bn=cfg.MODEL.PARTIAL_BN)
