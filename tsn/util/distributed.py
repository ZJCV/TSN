# -*- coding: utf-8 -*-

"""
@date: 2020/9/17 下午2:13
@file: distributed.py
@author: zj
@description: 
"""

import os
import numpy as np
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, gpus, backend='nccl'):
    # initialize the process group
    dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available() and gpus == 1:
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup():
    dist.destroy_process_group()


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
