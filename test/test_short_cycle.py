# -*- coding: utf-8 -*-

"""
@date: 2020/11/16 下午8:23
@file: test_short_cycle.py
@author: zj
@description: 
"""

from tsn.config import cfg
from torch.utils.data import SequentialSampler

from tsn.data.datasets.build import build_dataset
from tsn.data.transforms.build import build_transform

from tsn.data.samplers.short_cycle_batch_sampler import ShortCycleBatchSampler


def main():
    is_train = True
    transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, is_train=is_train)

    sampler = SequentialSampler(dataset)
    cfg.SAMPLER.MULTIGRID.DEFAULT_S = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
    sampler = ShortCycleBatchSampler(sampler, cfg.DATALOADER.TRAIN_BATCH_SIZE, False, cfg)

    print('batch_size:', cfg.DATALOADER.TRAIN_BATCH_SIZE)

    for i, idxs in enumerate(sampler):
        print(idxs)
        print(len(idxs))
        if i > 3:
            break

    print(len(sampler))


if __name__ == '__main__':
    main()
