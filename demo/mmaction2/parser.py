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


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--video_path', type=str, default="", help='video path, 0 presents camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    args = parser.parse_args()
    return args


def load_config(args):
    cfg.merge_from_file(args.config)
    cfg.MODEL.PRETRAINED = args.checkpoint

    cfg.freeze()
    return cfg
