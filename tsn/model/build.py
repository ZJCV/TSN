# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

from . import registry
from .tsn import TSN
from .criterions.crossentropy import build_crossentropy
from tsn.util.checkpoint import CheckPointer


def build_model(cfg, logger):
    model = TSN(cfg)

    if cfg.MODEL.PRETRAINED != "":
        if logger:
            logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model, logger=logger)
        checkpointer.load(cfg.MODEL.PRETRAINED)

    return model


def build_criterion(cfg):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg)
