# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:31
@file: misc.py
@author: zj
@description: 
"""

import os
import torch.multiprocessing as mp


def launch_job(cfg, func):
    gpus = cfg.NUM_GPUS
    if gpus > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '17928'
        mp.spawn(func, nprocs=gpus, args=(cfg,))
    else:
        func(gpu_id=0, cfg=cfg)
