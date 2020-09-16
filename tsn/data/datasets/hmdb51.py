# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import os
import numpy as np
from .base_dataset import BaseDataset

classes = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb',
           'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'eat',
           'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit',
           'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
           'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike',
           'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
           'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault',
           'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk',
           'throw', 'turn', 'walk', 'wave']


class HMDB51(BaseDataset):

    def __init__(self, data_dir, annotation_dir, modality="RGB", num_segs=3, split=1, train=True, transform=None):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(HMDB51, self).__init__(data_dir, modality=modality, num_segs=num_segs, transform=transform)

        annotation_list = list()
        if train:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{split}_rawframes.txt')
            if not os.path.isfile(annotation_path):
                raise ValueError(f'{annotation_path}不是文件路径')

            annotation_list.append(annotation_path)
        else:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{split}_rawframes.txt')
            if not os.path.isfile(annotation_path):
                raise ValueError(f'{annotation_path}不是文件路径')

            annotation_list.append(annotation_path)

        self.update(annotation_list)
        self.update_class(classes)
