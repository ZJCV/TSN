# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

from .augmentation import \
    Compose, \
    ScaleJitter, \
    RandomHorizontalFlip, \
    ColorJitter, \
    RandomRotation, \
    RandomCrop, \
    CenterCrop, \
    ToTensor, \
    Normalize, \
    RandomErasing, \
    Resize, \
    ThreeCrop


def build_transform(cfg, is_train=True):
    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD

    aug_list = list()
    if is_train:
        min, max = cfg.TRANSFORM.TRAIN.SCALE_JITTER
        assert max > 0 and min > 0 and max > min
        aug_list.append(ScaleJitter(min, max))
        if cfg.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP:
            aug_list.append(RandomHorizontalFlip())
        if cfg.TRANSFORM.TRAIN.COLOR_JITTER is not None:
            brightness, contrast, saturation, hue = cfg.TRANSFORM.TRAIN.COLOR_JITTER
            aug_list.append(
                ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        if cfg.TRANSFORM.TRAIN.RANDOM_ROTATION > 0:
            random_rotation = cfg.TRANSFORM.TRAIN.RANDOM_ROTATION
            aug_list.append(RandomRotation(random_rotation))
        if cfg.TRANSFORM.TRAIN.RANDOM_CROP:
            crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
            aug_list.append(RandomCrop(crop_size))
        if cfg.TRANSFORM.TRAIN.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
            aug_list.append(CenterCrop(crop_size))

        aug_list.append(ToTensor())
        aug_list.append(Normalize(MEAN, STD))

        if cfg.TRANSFORM.TRAIN.RANDOM_ERASING:
            aug_list.append(RandomErasing())
    else:
        shorter_side = cfg.TRANSFORM.TEST.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(Resize(shorter_side))
        if cfg.TRANSFORM.TEST.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
            aug_list.append(CenterCrop(crop_size))
        if cfg.TRANSFORM.TEST.THREE_CROP:
            crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
            aug_list.append(ThreeCrop(crop_size))

        aug_list.append(ToTensor())
        aug_list.append(Normalize(MEAN, STD))

    transform = Compose(aug_list)
    return transform
