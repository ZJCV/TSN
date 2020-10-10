# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import os

from .base_dataset import VideoRecord, BaseDataset

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

    def __init__(self,
                 *args,
                 split=1,
                 **kwargs):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(HMDB51, self).__init__(*args, **kwargs)

        self.split = split
        self.start_index = 0
        self.img_prefix = 'img_'

        self._update_video(self.annotation_dir, is_train=self.is_train)
        self._update_class()
        self._sample_frames()

    def _update_video(self, annotation_dir, is_train=True):
        if is_train:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{self.split}_rawframes.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{self.split}_rawframes.txt')

        if not os.path.isfile(annotation_path):
            raise ValueError(f'{annotation_path}不是文件路径')

        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(annotation_path)]

    def _update_class(self):
        self.classes = classes
