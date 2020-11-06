# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午8:00
@file: normalize.py
@author: zj
@description: 
"""

import torch
from opencv_transforms import transforms
from opencv_transforms import functional as F


class Normalize(transforms.Normalize):

    def __init__(self, mean, std):
        super().__init__(mean, std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if len(tensor.shape) == 4:
            return torch.stack([F.normalize(crop, self.mean, self.std) for crop in tensor])
        else:
            return super().__call__(tensor)

    def __repr__(self):
        return super().__repr__()
