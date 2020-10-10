# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午11:05
@file: base_dataset.py
@author: zj
@description: 
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .clipsample import SegmentedSample, DenseSample


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class BaseDataset(Dataset):

    def __init__(self,
                 data_dir,
                 annotation_dir,
                 modality="RGB",
                 sample_strategy='SegSample',
                 clip_len=1,
                 frame_interval=1,
                 num_clips=3,
                 is_train=True,
                 transform=None):
        assert isinstance(modality, str) and modality in ('RGB', 'RGBDiff')
        assert isinstance(sample_strategy, str) and sample_strategy in ('SegSample', 'DenseSample')

        if modality == 'RGB':
            assert clip_len == 1
        elif modality == 'RGBDiff':
            assert clip_len == 5
            # Diff needs one more image to calculate diff
            clip_len += 1
        else:
            raise ValueError(f'{self.modality} does not exist')

        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.modality = modality
        self.sample_strategy = sample_strategy
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.is_train = is_train
        self.transform = transform

        self.video_list = None
        self.cate_list = None
        self.img_num_list = None
        self.sampler = None
        # RawFrames下标从0开始，比如UCF101/HMDB51，也有用1开始，比如JESTER
        self.start_index = 0
        # RawFrames图像命令前缀，比如UCF101/HMDB51使用img_，JESTER没有
        self.img_prefix = 'img_'

    def _update_video(self, annotation_dir, is_train=True):
        pass

    def _update_class(self):
        pass

    def _sample_frames(self):
        if self.sample_strategy == 'SegSample':
            self.clip_sample = SegmentedSample(self.clip_len,
                                           self.frame_interval,
                                           self.num_clips,
                                           is_train=self.is_train,
                                           start_index=self.start_index)
        elif self.sample_strategy == 'DenseSample':
            self.clip_sample = DenseSample(self.clip_len,
                                       self.frame_interval,
                                       self.num_clips,
                                       is_train=self.is_train,
                                       start_index=self.start_index)
        else:
            raise ValueError(f'{self.sample_strategy} does not exist')

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧，则返回(T, C, H, W)，其中T表示num_segs
        """
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label

        clip_offsets = self.clip_sample(record.num_frames)

        video_path = os.path.join(self.data_dir, record.path)
        image_list = list()
        for offset in clip_offsets:
            if 'RGB' == self.modality:
                img_path = os.path.join(video_path, '{}{:0>5d}.jpg'.format(self.img_prefix, offset))
                img = np.array(Image.open(img_path))

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                for clip in range(self.clip_len):
                    img_path = os.path.join(video_path, '{}{:0>5d}.jpg'.format(self.img_prefix, offset + clip))
                    img = np.array(Image.open(img_path))

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_len)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    image_list.append(img)
        # [T, C, H, W] -> [C, T, H, W]
        image = torch.stack(image_list).transpose(0, 1)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)
