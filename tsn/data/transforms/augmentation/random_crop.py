# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:32
@file: random_crop.py
@author: zj
@description: 
"""

from opencv_transforms import transforms


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def __call__(self, img):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()
