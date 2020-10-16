# -*- coding: utf-8 -*-

"""
@date: 2020/9/17 下午2:13
@file: distributed.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.distributed as dist


def setup(rank, world_size, backend='nccl', seed=0):
    if world_size > 1:
        # initialize the process group
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    if get_world_size() > 1:
        dist.destroy_process_group()


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_device(gpu_id=None):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif gpu_id is None:
        device = torch.cuda.current_device()
    else:
        device = torch.device(f'cuda:{gpu_id}')
    return device


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
