# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets.build import build_dataset
from .samplers import IterationBasedBatchSampler
from .transforms.build import build_transform
import tsn.util.distributed as du


def build_dataloader(cfg,
                     is_train=True,
                     start_iter=0):
    transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, is_train=is_train)

    if is_train:
        batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE

        world_size = du.get_world_size()
        rank = du.get_rank()
        if world_size != 1 and rank == 0:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            # 训练阶段使用随机采样器
            sampler = torch.utils.data.RandomSampler(dataset)
    else:
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    if is_train:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset, num_workers=cfg.DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                             pin_memory=True)

    return data_loader
