# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午7:39
@file: __init__.py.py
@author: zj
@description: 
"""

"""
参考[Why torchvision doesn’t use opencv?](https://discuss.pytorch.org/t/why-torchvision-doesnt-use-opencv/24311)

使用[jbohnslav/opencv_transforms](https://github.com/jbohnslav/opencv_transforms)替代torchvision transforms实现
"""

from .compose import Compose

from .to_pil_image import ToPILImage
from .to_tensor import ToTensor
from .normalize import Normalize

from .resize import Resize
from .random_rotation import RandomRotation
from .random_erasing import RandomErasing
from .random_horizontal_flip import RandomHorizontalFlip

from .color_jitter import ColorJitter
from .scale_jitter import ScaleJitter

from .center_crop import CenterCrop
from .random_crop import RandomCrop
from .three_crop import ThreeCrop
