# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.model import registry

from .tsn_recognizer import TSNRecognizer


def build_recognizer(cfg):
    return registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg)
