# -*- coding: utf-8 -*-

"""
@date: 2020/9/25 下午4:53
@file: resize.py
@author: zj
@description:
"""

import cv2
from opencv_transforms import transforms


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()
