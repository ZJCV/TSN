# -*- coding: utf-8 -*-

"""
@date: 2020/11/21 下午7:24
@file: test_tsn_head.py
@author: zj
@description: 
"""

import torch
from tsn.model.heads.tsn_head import TSNHead

from tsn.config import cfg


def test_resnet_head():
    data = torch.randn(1, 2048, 7, 7)

    feature_dims = 2048
    num_classes = 1000
    cfg.MODEL.HEAD.FEATURE_DIMS = feature_dims
    cfg.MODEL.HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.HEAD.DROPOUT = 0
    model = TSNHead(cfg)

    outputs = model(data)
    print(outputs.shape)

    assert outputs.shape == (1, 1000)


if __name__ == '__main__':
    test_resnet_head()
