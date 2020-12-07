# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:49
@file: model.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    _C.MODEL = CN()
    _C.MODEL.NAME = "TSN"
    _C.MODEL.PRETRAINED = ""
    _C.MODEL.SYNC_BN = False

    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = 'ResNet50'
    _C.MODEL.BACKBONE.PARTIAL_BN = False
    _C.MODEL.BACKBONE.TORCHVISION_PRETRAINED = False
    _C.MODEL.BACKBONE.ZERO_INIT_RESIDUAL = False

    _C.MODEL.HEAD = CN()
    _C.MODEL.HEAD.NAME = 'TSNHead'
    _C.MODEL.HEAD.FEATURE_DIMS = 2048
    _C.MODEL.HEAD.DROPOUT = 0.0
    _C.MODEL.HEAD.NUM_CLASSES = 51

    _C.MODEL.RECOGNIZER = CN()
    _C.MODEL.RECOGNIZER.NAME = 'TSNRecognizer'

    _C.MODEL.CONSENSU = CN()
    _C.MODEL.CONSENSU.NAME = 'AvgConsensus'

    _C.MODEL.CRITERION = CN()
    _C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'
