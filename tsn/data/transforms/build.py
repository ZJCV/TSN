# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms
from .random_resize import RandomResize


def build_transform(cfg, train=True):
    min, max = cfg.TRANSFORM.JITTER_SCALES
    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD

    if train:
        crop_size = cfg.TRANSFORM.TRAIN_CROP_SIZE
        transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomResize(min, max),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(10),
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
