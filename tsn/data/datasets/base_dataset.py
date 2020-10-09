# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午11:05
@file: base_dataset.py
@author: zj
@description: 
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


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
        # RawFrames下标从0开始，比如UCF101/HMDB51，也有用1开始，比如JESTER
        self.base_index = 0
        # RawFrames图像命令前缀，比如UCF101/HMDB51使用img_，JESTER没有
        self.img_prefix = 'img_'

        self._update_video(self.annotation_dir, is_train=self.is_train)
        self._update_class()

    def _update_video(self, annotation_dir, is_train=True):
        pass

    def _update_class(self):
        pass

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.is_train:
            clip_offsets = self._get_train_clips(num_frames)
        else:
            clip_offsets = self._get_test_clips(num_frames)

        return clip_offsets + self.base_index

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧，则返回(T, C, H, W)，其中T表示num_segs
        """
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label

        clip_offsets = self._sample_clips(record.num_frames)

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
