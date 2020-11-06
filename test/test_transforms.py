# -*- coding: utf-8 -*-

"""
@date: 2020/11/5 下午7:23
@file: test_transforms.py
@author: zj
@description: 
"""

import numpy as np

import time
from tsn.data.transforms.build import build_transform
from tsn.config import cfg


def test_transforms():
    img = np.random.randn(224, 224, 3).astype(np.uint8)

    test_transform = build_transform(cfg, is_train=False)
    start = time.time()
    res = test_transform(img)
    end = time.time()
    print(res.shape)
    print('process time: {}'.format(end - start))

    train_transform = build_transform(cfg, is_train=True)
    start = time.time()
    res = train_transform(img)
    end = time.time()
    print(res.shape)
    print('process time: {}'.format(end - start))


if __name__ == '__main__':
    test_transforms()
