# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午10:19
@file: compose.py
@author: zj
@description: 
"""

from opencv_transforms import transforms


class Compose(transforms.Compose):

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()
