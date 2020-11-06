# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:31
@file: random_rotation.py
@author: zj
@description: 
"""

from opencv_transforms import transforms


class RandomRotation(transforms.RandomRotation):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        super().__init__(degrees, resample, expand, center)

    def __call__(self, img):
        return super().__call__(img)
