# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import os

from .base_dataset import BaseDataset


class HMDB51(BaseDataset):

    def __init__(self, data_dir, annotation_dir, modality=("RGB"), num_seg=3, splits=(1,), train=True, transform=None):
        assert isinstance(splits, tuple) and len(splits) <= 3
        super(HMDB51, self).__init__(data_dir, modality=modality, num_seg=num_seg, transform=transform)

        annotation_list = list()
        if train:
            for split in splits:
                annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{split}_rawframes.txt')
                if not os.path.isfile(annotation_path):
                    raise ValueError(f'{annotation_path}不是文件路径')

                annotation_list.append(annotation_path)
        else:
            for split in splits:
                annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{split}_rawframes.txt')
                if not os.path.isfile(annotation_path):
                    raise ValueError(f'{annotation_path}不是文件路径')

                annotation_list.append(annotation_path)

        self.update(annotation_list)
