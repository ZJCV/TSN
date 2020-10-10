# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms
from .random_resize import RandomResize


def build_transform(cfg, is_train=True):
    min, max = cfg.TRANSFORM.JITTER_SCALES
    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD
    RANDOM_ROTATION = cfg.TRANSFORM.RANDOM_ROTATION
    brightness, contrast, saturation, hue = cfg.TRANSFORM.COLOR_JITTER

    if is_train:
        crop_size = cfg.TRANSFORM.TRAIN_CROP_SIZE
        transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomResize(min, max),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.RandomRotation(RANDOM_ROTATION),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing()
        ])
    else:
        crop_size = cfg.TRANSFORM.TEST_CROP_SIZE
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    return transform
