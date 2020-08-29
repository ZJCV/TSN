# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tsn.data.datasets.hmdb51 import HMDB51
from tsn.data.datasets.ucf101 import UCF101


def build_train_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing()
    ])

    return transform, None


def build_test_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform


def build_dataset():
    # data_dir = '/home/zj/zhonglian/mmaction2/data/hmdb51/rawframes'
    # annotation_dir = '/home/zj/zhonglian/mmaction2/data/hmdb51'
    data_dir = '/home/zj/zhonglian/mmaction2/data/ucf101/rawframes'
    annotation_dir = '/home/zj/zhonglian/mmaction2/data/ucf101'

    train_transform, _ = build_train_transform()
    test_transform = build_test_transform()

    train_dataset = UCF101(data_dir, annotation_dir, num_seg=3, split=1, modality=('RGB', 'RGBDiff'),
                           train=True, transform=train_transform)
    test_dataset = UCF101(data_dir, annotation_dir, num_seg=3, split=1, modality=('RGB', 'RGBDiff'),
                          train=False, transform=test_transform)

    return {'train': train_dataset, 'test': test_dataset}, {'train': len(train_dataset), 'test': len(test_dataset)}


def build_dataloader():
    data_sets, data_sizes = build_dataset()

    train_dataloader = DataLoader(data_sets['train'], batch_size=16, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(data_sets['test'], batch_size=16, shuffle=True, num_workers=8)

    return {'train': train_dataloader, 'test': test_dataloader}, data_sizes
