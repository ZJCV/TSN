# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from .resnet import resnet50


def build_backbone(name, partial_bn=False):
    if name == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        raise ValueError('no matching backbone exists')
    return model
