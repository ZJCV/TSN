# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: build.py
@author: zj
@description: 
"""

import torch.optim as optim


def build_optimizer(cfg, model):
    return optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    # return optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)


def build_lr_scheduler(cfg, optimizer):
    milestones = cfg.LR_SCHEDULER.MILESTONES
    gamma = cfg.LR_SCHEDULER.GAMMA

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
