# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import os
import torch
import argparse

from tsn.config import cfg
from tsn.data.build import build_dataloader
from tsn.model.build import build_model, build_criterion
from tsn.optim.build import build_optimizer, build_lr_scheduler
from tsn.engine.trainer import do_train
from tsn.engine.inference import do_evaluation
from tsn.util.checkpoint import CheckPointer
from tsn.util.logger import setup_logger
from tsn.util.collect_env import collect_env_info


def train(cfg, arguments, device):
    logger = setup_logger(cfg.TRAIN.NAME)

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT.DIR,
                                save_to_disk=True, logger=logger)
    if arguments['resume']:
        logger.info('resume ...')
        extra_checkpoint_data = checkpointer.load()
        arguments.update(extra_checkpoint_data)
    if cfg.LR_SCHEDULER.WARMUP:
        logger.info('warmup ...')
        if lr_scheduler.finished:
            optimizer.load_state_dict(lr_scheduler.after_scheduler.optimizer.state_dict())
        else:
            optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
        lr_scheduler.optimizer = optimizer
        lr_scheduler.after_scheduler.optimizer = optimizer

    data_loader = build_dataloader(cfg, train=True, start_iter=arguments['iteration'])
    model = do_train(cfg, arguments,
                     data_loader, model, criterion, optimizer, lr_scheduler, checkpointer,
                     device, logger)
    return model


def main():
    parser = argparse.ArgumentParser(description='TSN Training With PyTorch')
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--use_save', default=True, type=bool)
    parser.add_argument('--eval_step', default=2500, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_eval', default=True, type=bool)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_tensorboard', default=True, type=bool)

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    arguments = {"iteration": 0}
    arguments['log_step'] = args.log_step
    arguments['save_step'] = args.save_step
    arguments['use_save'] = args.use_save
    arguments['eval_step'] = args.eval_step
    arguments['use_eval'] = args.use_eval
    arguments['resume'] = args.resume
    arguments['use_tensorboard'] = args.use_tensorboard

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(cfg.OUTPUT.DIR):
        os.makedirs(cfg.OUTPUT.DIR)

    logger = setup_logger("TSN", save_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train(cfg, arguments, device)

    logger.info('Start final evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model, device)


if __name__ == '__main__':
    main()
