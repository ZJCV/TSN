# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午10:07
@file: scale_jitter.py
@author: zj
@description: 
"""

import cv2
import numpy as np
from opencv_transforms import transforms
from opencv_transforms import functional as F


class ScaleJitter(transforms.Resize):

    def __init__(self, min, max, interpolation=cv2.INTER_LINEAR):
        assert min < max
        self.min = min
        self.max = max
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        size = np.random.randint(self.min, self.max)
        return F.resize(img, (size, size), self.interpolation)

    def __repr__(self):
        return super().__repr__()
