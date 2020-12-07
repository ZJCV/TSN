# -*- coding: utf-8 -*-

"""
@date: 2020/11/25 下午6:52
@file: dataset.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # DataSets
    # ---------------------------------------------------------------------------- #
    _C.DATASETS = CN()
    _C.DATASETS.TYPE = 'RawFrame'
    _C.DATASETS.MODALITY = 'RGB'
    _C.DATASETS.SAMPLE_STRATEGY = 'SegSample'
    _C.DATASETS.CLIP_LEN = 1
    _C.DATASETS.FRAME_INTERVAL = 1
    _C.DATASETS.NUM_CLIPS = 3
    # for densesample test
    _C.DATASETS.NUM_SAMPLE_POSITIONS = 10
    # for vidoe decode
    # Enable multi thread decoding.
    _C.DATASETS.ENABLE_MULTI_THREAD_DECODE = False
    # Decoding backend, options include `pyav` or `torchvision`
    _C.DATASETS.DECODING_BACKEND = "pyav"
    # train
    _C.DATASETS.TRAIN = CN()
    _C.DATASETS.TRAIN.NAME = 'HMDB51'
    _C.DATASETS.TRAIN.DATA_DIR = 'data/hmdb51/rawframes'
    _C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/hmdb51'
    # for hmdb51 and ucf101
    _C.DATASETS.TRAIN.SPLIT = 1
    # test
    _C.DATASETS.TEST = CN()
    _C.DATASETS.TEST.NAME = 'HMDB51'
    _C.DATASETS.TEST.DATA_DIR = 'data/hmdb51/rawframes'
    _C.DATASETS.TEST.ANNOTATION_DIR = 'data/hmdb51'
    # for hmdb51 and ucf101
    _C.DATASETS.TEST.SPLIT = 1
