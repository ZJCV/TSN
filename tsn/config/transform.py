# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:51
@file: transform.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Transform
    # ---------------------------------------------------------------------------- #
    _C.TRANSFORM = CN()
    _C.TRANSFORM.MEAN = (0.45, 0.45, 0.45)  # (0.485, 0.456, 0.406)
    _C.TRANSFORM.STD = (0.225, 0.225, 0.225)  # (0.229, 0.224, 0.225)

    _C.TRANSFORM.TRAIN = CN()
    _C.TRANSFORM.TRAIN.SCALE_JITTER = (256, 320)
    _C.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP = True
    # (brightness, contrast, saturation, hue)
    _C.TRANSFORM.TRAIN.COLOR_JITTER = (0.1, 0.1, 0.1, 0.1)
    _C.TRANSFORM.TRAIN.RANDOM_ROTATION = 10
    _C.TRANSFORM.TRAIN.RANDOM_CROP = True
    _C.TRANSFORM.TRAIN.CENTER_CROP = False
    _C.TRANSFORM.TRAIN.TRAIN_CROP_SIZE = 224
    _C.TRANSFORM.TRAIN.RANDOM_ERASING = True

    _C.TRANSFORM.TEST = CN()
    _C.TRANSFORM.TEST.SHORTER_SIDE = 256
    _C.TRANSFORM.TEST.CENTER_CROP = True
    _C.TRANSFORM.TEST.THREE_CROP = False
    _C.TRANSFORM.TEST.TEST_CROP_SIZE = 256
