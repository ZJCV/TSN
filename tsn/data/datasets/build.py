# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from .hmdb51 import HMDB51
from .ucf101 import UCF101


def build_dataset(cfg, transform=None, is_train=True):
    dataset_name = cfg.DATASETS.TRAIN.NAME if is_train else cfg.DATASETS.TEST.NAME
    splits = cfg.DATASETS.TRAIN.SPLITS if is_train else cfg.DATASETS.TEST.SPLITS
    data_dir = cfg.DATASETS.TRAIN.DATA_DIR if is_train else cfg.DATASETS.TEST.DATA_DIR
    annotation_dir = cfg.DATASETS.TRAIN.ANNOTATION_DIR if is_train else cfg.DATASETS.TEST.ANNOTATION_DIR

    modality = cfg.DATASETS.MODALITY

    if dataset_name == 'HMDB51':
        dataset = HMDB51(data_dir, annotation_dir, train=is_train, modality=modality, splits=splits,
                         transform=transform)
    elif dataset_name == 'UCF101':
        dataset = UCF101(data_dir, annotation_dir, train=is_train, modality=modality, splits=splits,
                         transform=transform)

    return dataset