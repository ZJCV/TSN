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

"""
opencv_transform vs. torchvision transform

torchvhsion transform 测试
$ python test/test_transforms2.py 
torch.Size([3, 256, 256])
process time: 0.015646696090698242
torch.Size([3, 224, 224])
process time: 0.042925119400024414
$ python test/test_transforms2.py 
torch.Size([3, 256, 256])
process time: 0.020355701446533203
torch.Size([3, 224, 224])
process time: 0.021254539489746094
$ python test/test_transforms2.py 
torch.Size([3, 256, 256])
process time: 0.015828847885131836
torch.Size([3, 224, 224])
process time: 0.03472018241882324

opencv_transform 测试
$ python test/test_transforms.py 
torch.Size([3, 256, 256])
process time: 0.011228799819946289
torch.Size([3, 224, 224])
process time: 0.02078700065612793
$ python test/test_transforms.py 
torch.Size([3, 256, 256])
process time: 0.018286943435668945
torch.Size([3, 224, 224])
process time: 0.02521538734436035
$ python test/test_transforms.py 
torch.Size([3, 256, 256])
process time: 0.00945901870727539
torch.Size([3, 224, 224])
process time: 0.02595067024230957
"""


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
