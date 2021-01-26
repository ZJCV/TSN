# -*- coding: utf-8 -*-

"""
@date: 2021/1/26 上午10:51
@file: misc.py
@author: zj
@description: 
"""

import cv2

def get_cap(cfg):
    assert (
            cfg.DEMO.WEBCAM > -1 or cfg.DEMO.INPUT_VIDEO != ""
    ), "Must specify a data source as input."

    source = (
        cfg.DEMO.WEBCAM if cfg.DEMO.WEBCAM > -1 else cfg.DEMO.INPUT_VIDEO
    )
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise IOError("Video {} cannot be opened".format(source))

    return cap


def get_output_file(cfg, cap):
    display_width = int(cfg.DEMO.DISPLAY_WIDTH)
    display_height = int(cfg.DEMO.DISPLAY_HEIGHT)

    if display_width > 0 and display_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
    else:
        display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cfg.DEMO.OUTPUT_FPS == -1:
        output_fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        output_fps = cfg.DEMO.OUTPUT_FPS

    if cfg.DEMO.OUTPUT_FILE != "":
        output_file = cv2.VideoWriter(
            filename=cfg.DEMO.OUTPUT_FILE,
            fourcc=cv2.VideoWriter_fourcc(*'avc1'),
            fps=float(output_fps),
            frameSize=(display_width, display_height),
            isColor=True,
        )
    else:
        output_file = None

    return output_file, output_fps


def get_label(cfg):
    if cfg.DEMO.LABEL_FILE_PATH != "":
        with open(cfg.DEMO.LABEL_FILE_PATH, 'r') as f:
            label = [line.strip().split(' ')[1] for line in f]
        return label
    else:
        raise ValueError(f'{cfg.DEMO.LABEL_FILE_PATH} does not exist')
