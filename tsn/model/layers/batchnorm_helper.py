# -*- coding: utf-8 -*-

"""
@date: 2020/9/23 下午2:35
@file: batchnorm_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def convert_sync_bn(model, process_group):
    sync_bn_module = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    return sync_bn_module
