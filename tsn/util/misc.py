# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:31
@file: misc.py
@author: zj
@description: 
"""

import os
import inspect
import torch.multiprocessing as mp


def launch_job(cfg, func):
    gpus = cfg.NUM_GPUS
    if gpus > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '17928'
        mp.spawn(func, nprocs=gpus, args=(cfg,))
    else:
        func(gpu_id=0, cfg=cfg)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
