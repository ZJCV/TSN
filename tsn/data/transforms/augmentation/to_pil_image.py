# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:28
@file: to_pil_image.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


class ToPILImage(transforms.ToPILImage):

    def __init__(self, mode=None):
        super().__init__(mode)

    def __call__(self, pic):
        return super().__call__(pic)

    def __repr__(self):
        return super().__repr__()
