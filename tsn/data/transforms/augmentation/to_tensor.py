# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午7:49
@file: to_tensor.py
@author: zj
@description: 
"""

import torch
from opencv_transforms import transforms
from opencv_transforms import functional as F


class ToTensor(transforms.ToTensor):

    def __call__(self, pic):
        """
        Args:
            pic : Image to be converted to tensor.
            1. if pic == (PIL Image or numpy.ndarray)
            2. if pic == tuple

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, tuple):
            return torch.stack([F.to_tensor(crop) for crop in pic])
        else:
            return super().__call__(pic)

    def __repr__(self):
        return super().__repr__()
