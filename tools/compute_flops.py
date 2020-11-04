# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午2:06
@file: compute_flops.py
@author: zj
@description: 
"""

import torch

from tsn.util.distributed import get_device, get_local_rank
from tsn.util.metrics import compute_num_flops
from tsn.config import cfg
from tsn.model.recognizers.build import build_recognizer


def main():
    # cfg.merge_from_file('configs/tsn_r50_ucf101_rgb_raw_dense_1x16x4.yaml')
    cfg.merge_from_file('configs/tsn_r50_ucf101_rgb_raw_seg_1x1x3.yaml')

    device = get_device(local_rank=get_local_rank())
    model = build_recognizer(cfg, device)
    data = torch.randn((1, 3, 3, 256, 256)).to(device=device, non_blocking=True)

    GFlops, params_size = compute_num_flops(model, data)
    print(f'device: {device}')
    print(f'GFlops: {GFlops}')
    print(f'Params Size: {params_size}')


if __name__ == '__main__':
    main()
