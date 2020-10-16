# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import torch

from tsn.data.build import build_dataloader
from tsn.model.build import build_model, build_criterion
from tsn.optim.build import build_optimizer, build_lr_scheduler
from tsn.engine.trainer import do_train
from tsn.util.checkpoint import CheckPointer
from tsn.util import logging
from tsn.util.collect_env import collect_env_info
from tsn.util.parser import parse_train_args, load_config
from tsn.util.misc import launch_job
from tsn.util.distributed import setup, cleanup, is_master_proc, get_device


def train(gpu_id, cfg):
    rank = cfg.RANK * cfg.NUM_GPUS + gpu_id
    world_size = cfg.WORLD_SIZE
    setup(rank, world_size, seed=cfg.RNG_SEED)

    logger = logging.setup_logging()
    arguments = {"iteration": 0, 'gpu_id': gpu_id}

    torch.cuda.set_device(gpu_id)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    model = build_model(cfg, gpu_id)
    criterion = build_criterion(cfg, gpu_id)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT.DIR,
                                save_to_disk=True, logger=logger)
    if cfg.TRAIN.RESUME:
        if is_master_proc():
            logger.info('resume ...')
        extra_checkpoint_data = checkpointer.load(map_location=map_location, rank=rank)
        if extra_checkpoint_data != dict():
            arguments['iteration'] = extra_checkpoint_data['iteration']
            if cfg.LR_SCHEDULER.IS_WARMUP:
                if is_master_proc():
                    logger.info('warmup ...')
                if lr_scheduler.finished:
                    optimizer.load_state_dict(lr_scheduler.after_scheduler.optimizer.state_dict())
                else:
                    optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
                lr_scheduler.optimizer = optimizer
                lr_scheduler.after_scheduler.optimizer = optimizer

    data_loader = build_dataloader(cfg, is_train=True, start_iter=arguments['iteration'])

    do_train(cfg, arguments,
             data_loader, model, criterion, optimizer, lr_scheduler,
             checkpointer)
    cleanup()


def main():
    args = parse_train_args()
    cfg = load_config(args)

    logger = logging.setup_logging(output_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg, train)


if __name__ == '__main__':
    main()
