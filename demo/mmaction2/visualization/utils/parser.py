# -*- coding: utf-8 -*-

"""
@date: 2020/10/4 下午3:09
@file: parser.py
@author: zj
@description: 
"""

import argparse
from demo.mmaction2.visualization.configs import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam visualization')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def load_config(args):
    cfg.merge_from_file(args.config)

    cfg.freeze()
    return cfg
