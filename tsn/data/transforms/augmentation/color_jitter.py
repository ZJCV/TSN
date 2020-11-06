# -*- coding: utf-8 -*-

"""
@date: 2020/11/6 上午9:30
@file: color_jitter.py
@author: zj
@description: 
"""

import numpy as np
import torchvision.transforms as transforms

from .to_pil_image import ToPILImage

"""
refer to [jbohnslav/opencv_transforms](https://github.com/jbohnslav/opencv_transforms)
Speed up augmentation in saturation and hue. Currently, fastest way is to convert to a PIL image, perform same augmentation as Torchvision, then convert back to np.ndarray
"""


class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

        self.to_pil_image = ToPILImage()

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        return super()._check_input(value, name, center, bound, clip_first_on_zero)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        return super().get_params(brightness, contrast, saturation, hue)

    def forward(self, img):
        """
        先转换成PIL Image，再使用torchvision进行color jitter，然后转换成numpy ndarray
        :param img:
        :return:
        """
        if isinstance(img, np.ndarray):
            img = self.to_pil_image(img)
        img = super().forward(img)
        return np.array(img)

    def __repr__(self):
        return super().__repr__()
