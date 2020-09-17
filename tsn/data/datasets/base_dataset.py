# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午11:05
@file: base_dataset.py
@author: zj
@description: 
"""

import cv2
from PIL import Image
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from tsn.util.image import rgbdiff


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

    def __init__(self, data_dir, train=True, modality="RGB", num_segs=3, transform=None):
        assert isinstance(modality, str) and modality in ('RGB', 'RGBDiff')

        self.data_dir = data_dir
        self.transform = transform
        self.num_segs = num_segs
        self.modality = modality
        self.train = train

        self.video_list = None
        self.cate_list = None
        self.img_num_list = None

        # 每个片段采集的帧数
        if self.modality == 'RGB':
            self.clip_length = 1
        if self.modality == 'RGBDiff':
            self.clip_length = 5 + 1  # Diff needs one more image to calculate diff

    def _update(self, annotation_path):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(annotation_path)]

    def _update_class(self, classes):
        self.classes = classes

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.clip_length + 1) // self.num_segs
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segs)), average_duration) + \
                      np.random.randint(average_duration, size=self.num_segs)
        elif record.num_frames > self.num_segs:
            offsets = np.sort(np.random.randint(record.num_frames - self.clip_length + 1, size=self.num_segs))
        else:
            offsets = np.zeros((self.num_segs,))
        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.clip_length + 1) / float(self.num_segs)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segs)])

        return offsets

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧，则返回(T, C, H, W)，其中T表示num_segs
        """
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label

        if self.train:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        video_path = os.path.join(self.data_dir, record.path)
        image_list = list()
        for num in segment_indices:
            if 'RGB' == self.modality:
                image_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num))
                img = cv2.imread(image_path)

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                for clip in range(self.clip_length):
                    img_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num + clip))
                    img = np.array(Image.open(img_path))

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_length)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    image_list.append(img)
        image = torch.stack(image_list)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)
