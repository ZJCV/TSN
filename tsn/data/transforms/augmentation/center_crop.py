# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:33
@file: center_crop.py
@author: zj
@description: 
"""

from opencv_transforms import transforms


class CenterCrop(transforms.CenterCrop):

    def __init__(self, size):
        super().__init__(size)

    def __call__(self, img):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()
