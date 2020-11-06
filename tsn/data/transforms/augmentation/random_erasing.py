# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:41
@file: random_erasing.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


class RandomErasing(transforms.RandomErasing):

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p, scale, ratio, value, inplace)

    def __call__(self, img):
        return super().__call__(img)
