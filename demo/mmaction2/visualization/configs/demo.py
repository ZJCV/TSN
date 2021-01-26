# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午9:33
@file: slowfast.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_config(_C):
    # ---------------------------------------------------------------------------- #
    # Demo options
    # ---------------------------------------------------------------------------- #
    _C.DEMO = CN()
    # ---------------------------------------------------------------------------- #
    # show_results
    # ---------------------------------------------------------------------------- #
    # Specify a camera device as input. This will be prioritized
    # over input video if set.
    # If -1, use input video instead.
    _C.DEMO.WEBCAM = -1
    # Path to input video for visualization.
    _C.DEMO.INPUT_VIDEO = ""
    # Custom width for reading input video data.
    _C.DEMO.DISPLAY_WIDTH = 0
    # Custom height for reading input video data.
    _C.DEMO.DISPLAY_HEIGHT = 0
    # If specified, the visualized outputs will be written this a video file of
    # this path. Otherwise, the visualized outputs will be displayed in a window.
    _C.DEMO.OUTPUT_FILE = ""
    # Frames per second rate for writing to output video file.
    # If not set (-1), use fps rate from input file.
    _C.DEMO.OUTPUT_FPS = -1
    # ---------------------------------------------------------------------------- #
    # inference
    # ---------------------------------------------------------------------------- #
    # Path to a json file providing class_name - id mapping
    # in the format {"class_name1": id1, "class_name2": id2, ...}.
    _C.DEMO.LABEL_FILE_PATH = ""
    # 平均多少轮检测成绩
    # number of latest clips to be averaged for prediction
    _C.DEMO.AVG_SIZE = 1
    # recognition score threshold
    _C.DEMO.THRESHOLD = 0.01
