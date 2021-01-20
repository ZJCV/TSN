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

    # Run model in DEMO mode.
    _C.DEMO.ENABLE = False

    # Path to a json file providing class_name - id mapping
    # in the format {"class_name1": id1, "class_name2": id2, ...}.
    _C.DEMO.LABEL_FILE_PATH = ""

    # Specify a camera device as input. This will be prioritized
    # over input video if set.
    # If -1, use input video instead.
    _C.DEMO.WEBCAM = -1

    # Path to input video for demo.
    _C.DEMO.INPUT_VIDEO = ""
    # Custom width for reading input video data.
    _C.DEMO.DISPLAY_WIDTH = 0
    # Custom height for reading input video data.
    _C.DEMO.DISPLAY_HEIGHT = 0
    # Path to Detectron2 object detection model configuration,
    # only used for detection tasks.
    _C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # Path to Detectron2 object detection model pre-trained weights.
    _C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    # Threshold for choosing predicted bounding boxes by Detectron2.
    _C.DEMO.DETECTRON2_THRESH = 0.9
    # Number of overlapping frames between 2 consecutive clips.
    # Increase this number for more frequent action predictions.
    # The number of overlapping frames cannot be larger than
    # half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
    _C.DEMO.BUFFER_SIZE = 0
    # If specified, the visualized outputs will be written this a video file of
    # this path. Otherwise, the visualized outputs will be displayed in a window.
    _C.DEMO.OUTPUT_FILE = ""
    # Frames per second rate for writing to output video file.
    # If not set (-1), use fps rate from input file.
    _C.DEMO.OUTPUT_FPS = -1
    # Input format from demo video reader ("RGB" or "BGR").
    _C.DEMO.INPUT_FORMAT = "BGR"
    # Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
    _C.DEMO.CLIP_VIS_SIZE = 10
    # Number of processes to run video visualizer.
    _C.DEMO.NUM_VIS_INSTANCES = 2

    # Path to pre-computed predicted boxes
    _C.DEMO.PREDS_BOXES = ""
    # Whether to run in with multi-threaded video reader.
    _C.DEMO.THREAD_ENABLE = False
    # Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
    # This is used for fast demo speed by reducing the prediction/visualiztion frequency.
    # If -1, take the most recent read clip for visualization. This mode is only supported
    # if `DEMO.THREAD_ENABLE` is set to True.
    _C.DEMO.NUM_CLIPS_SKIP = 0
    # Path to ground-truth boxes and labels (optional)
    _C.DEMO.GT_BOXES = ""
    # The starting second of the video w.r.t bounding boxes file.
    _C.DEMO.STARTING_SECOND = 900
    # Frames per second of the input video/folder of images.
    _C.DEMO.FPS = 30
    # Visualize with top-k predictions or predictions above certain threshold(s).
    # Option: {"thres", "top-k"}
    _C.DEMO.VIS_MODE = "thres"
    # Threshold for common class names.
    _C.DEMO.COMMON_CLASS_THRES = 0.7
    # Theshold for uncommon class names. This will not be
    # used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
    _C.DEMO.UNCOMMON_CLASS_THRES = 0.3
    # This is chosen based on distribution of examples in
    # each classes in AVA dataset.
    _C.DEMO.COMMON_CLASS_NAMES = [
        "watch (a person)",
        "talk to (e.g., self, a person, a group)",
        "listen to (a person)",
        "touch (an object)",
        "carry/hold (an object)",
        "walk",
        "sit",
        "lie/sleep",
        "bend/bow (at the waist)",
    ]
    # Slow-motion rate for the visualization. The visualized portions of the
    # video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
    _C.DEMO.SLOWMO = 1
