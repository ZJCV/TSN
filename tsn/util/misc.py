# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:31
@file: misc.py
@author: zj
@description: 
"""

import os
import torch.multiprocessing as mp


def launch_job(args, cfg, func):
    if args.gpus > 1:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '17928'
        mp.spawn(func, nprocs=args.gpus, args=(args, cfg))
    else:
        func(cfg=cfg)
