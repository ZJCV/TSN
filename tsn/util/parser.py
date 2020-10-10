# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:09
@file: parser.py
@author: zj
@description: 
"""

import os
import argparse
from tsn.config import cfg


def parse_train_args():
    parser = argparse.ArgumentParser(description='TSN Training With PyTorch')
    parser.add_argument("--config_file",
                        type=str,
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument('--log_step',
                        type=int,
                        default=10,
                        help='Print logs every log_step')

    parser.add_argument('--save_step',
                        type=int,
                        default=2500,
                        help='Save checkpoint every save_step')
    parser.add_argument('--stop_save',
                        default=False,
                        action='store_true')

    parser.add_argument('--eval_step',
                        type=int,
                        default=2500,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--stop_eval',
                        default=False,
                        action='store_true')

    parser.add_argument('--resume',
                        default=False,
                        action='store_true',
                        help='Resume training')
    parser.add_argument('--use_tensorboard',
                        type=int,
                        default=1)

    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of machines (default: 1)')
    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=1,
                        help='number of gpus per node')
    parser.add_argument('-nr',
                        '--nr',
                        type=int,
                        default=0,
                        help='ranking within the nodes')

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def parse_test_args():
    pass


def load_config(args):
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.gpus > 1:
        cfg.OPTIMIZER.LR *= args.gpus
        cfg.OPTIMIZER.WEIGHT_DECAY *= args.gpus
        cfg.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR *= args.gpus

    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT.DIR):
        os.makedirs(cfg.OUTPUT.DIR)

    return cfg
