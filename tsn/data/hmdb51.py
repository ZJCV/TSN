# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import cv2
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class HMDB51(Dataset):

    def __init__(self, data_dir, annotation_dir, num_seg=3, split=1, train=True, transform=None):
        if train:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{split}_rawframes.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{split}_rawframes.txt')

        if not os.path.isfile(annotation_path):
            raise ValueError(f'{annotation_path}不是文件路径')

        self.data_dir = data_dir
        self.transform = transform
        self.num_seg = num_seg

        video_list = list()
        img_num_list = list()
        cate_list = list()
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dir_name, img_num, cate = line.strip().split(' ')

                video_list.append(dir_name)
                img_num_list.append(int(img_num))
                cate_list.append(int(cate))
        self.video_list = video_list
        self.img_num_list = img_num_list
        self.cate_list = cate_list

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧
        :return: (T, C, H, W)，其中T表示num_seg
        """
        assert index < len(self.video_list)
        target = self.cate_list[index]

        num_list = random.sample(range(self.img_num_list[index]), self.num_seg)
        video_path = os.path.join(self.data_dir, self.video_list[index])

        image_list = list()
        for num in num_list:
            image_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num))
            img = cv2.imread(image_path)

            if self.transform:
                img = self.transform(img)
            image_list.append(img)
        image = torch.stack(image_list)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)