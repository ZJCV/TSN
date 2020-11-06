# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:29
@file: random_horizontal_flip.py
@author: zj
@description: 
"""

from opencv_transforms import transforms


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()
