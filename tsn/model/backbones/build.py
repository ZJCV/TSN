# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from .resnet.build_resnet import resnet50
from tsn.model import registry


def build_backbone(cfg, map_location=None):
    return registry.BACKBONE[cfg.MODEL.BACKBONE.NAME] \
        (pretrained=cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED,
         zero_init_residual=cfg.MODEL.BACKBONE.ZERO_INIT_RESIDUAL,
         partial_bn=cfg.MODEL.BACKBONE.PARTIAL_BN,
         map_location=map_location)
