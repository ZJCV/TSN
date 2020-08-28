# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import os
import torch

from tsn.data.build import build_dataloader
from tsn.model.build import build_model, build_criterion
from tsn.optim.build import build_optimizer, build_lr_scheduler
from tsn.engine.build import train_model
from tsn.util.checkpoint import CheckPointer

if __name__ == '__main__':
    epoches = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loaders, data_sizes = build_dataloader()

    criterion = build_criterion()
    model = build_model(num_classes=51).to(device)
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)

    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=output_dir,
                                save_to_disk=True, logger=None)

    train_model('MobileNet_v2', model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes, checkpointer,
                epoches=epoches, device=device)
