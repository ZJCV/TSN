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

    def __init__(self, data_dir, modality="RGB", num_segs=3, transform=None):
        assert isinstance(modality, str) and modality in ('RGB', 'RBGDiff')

        self.data_dir = data_dir
        self.transform = transform
        self.num_segs = num_segs
        self.modality = modality

        self.video_list = None
        self.cate_list = None
        self.img_num_list = None

    def update(self, annotation_path):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(annotation_path)]

    def update_class(self, classes):
        self.classes = classes

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧
        如果选择了输入模态为RGB或者RGBDiff,则返回(T, C, H, W)，其中T表示num_segs；
        如果输入模态为(RGB, RGBDiff)，则返回(T*2, C, H, W)
        """
        assert index < len(self.video_list)
        record = self.video_list[index]

        target = record.label

        # 视频帧数
        video_length = record.num_frames
        # 每一段帧数
        seg_length = int(video_length / self.num_segs)
        num_list = list()
        if 'RGBDiff' == self.modality:
            # 在每段中随机挑选一帧
            # 此处使用当前帧和下一帧进行Diff计算，在实际计算过程中，应该使用前一帧和当前帧进行Diff计算
            # 当然，当前实现也可以把下一帧看成是当前帧
            for i in range(self.num_segs):
                # 如果使用`RGBDiff`，需要采集前后两帧进行差分
                # random.randint(a, b) -> [a, b]
                num_list.append(random.randint(i * seg_length, (i + 1) * seg_length - 2))
        else:
            # 在每段中随机挑选一帧
            for i in range(self.num_segs):
                num_list.append(random.randint(i * seg_length, (i + 1) * seg_length - 1))
        video_path = os.path.join(self.data_dir, record.path)

        image_list = list()
        for num in num_list:
            if 'RGB' == self.modality:
                image_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num))
                img = cv2.imread(image_path)

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                img1_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num))
                # img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
                img1 = np.array(Image.open(img1_path))

                img2_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num + 1))
                # img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
                img2 = np.array(Image.open(img2_path))

                # print(img1.shape, img2.shape)
                img = rgbdiff(img1, img2)
                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
        image = torch.stack(image_list)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)
