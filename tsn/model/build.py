# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

from . import registry
from .recognizers.tsn_recognizer import TSNRecognizer
from .criterions.crossentropy import build_crossentropy


def build_model(cfg, map_location=None):
    return registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg, map_location=map_location)


def build_criterion(cfg):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg)
