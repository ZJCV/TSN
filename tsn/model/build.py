# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

from tsn.model.batchnorm_helper import simple_group_split, convert_sync_bn
import tsn.util.distributed as du
from tsn.util.checkpoint import CheckPointer
from tsn.util import logging
from . import registry
from .recognizers.tsn_recognizer import TSNRecognizer
from .criterions.crossentropy_loss import CrossEntropyLoss


def build_model(cfg, gpu_id):
    device = du.get_device(gpu_id)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % du.get_rank()}
    model = registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg, map_location=map_location).to(device=device)

    world_size = du.get_world_size()
    rank = du.get_rank()
    if cfg.MODEL.SYNC_BN and world_size > 1:
        process_group = simple_group_split(world_size, rank, 1)
        convert_sync_bn(model, process_group, gpu_id=gpu_id)
    if cfg.MODEL.PRETRAINED != "":
        logger = logging.setup_logging(__name__)
        logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model, logger=logger)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=map_location, rank=rank)

    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model


def build_criterion(cfg, gpu_id=None):
    device = du.get_device(gpu_id)
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
