# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午3:39
@file: video_provider.py
@author: zj
@description: 
"""

import cv2
from ..task_info import TaskInfo


class VideoProvider:
    """
    视频流读取
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
            tsn/config/visualization.py
        """
        assert (
                cfg.VISUALIZATION.WEBCAM > -1 or cfg.VISUALIZATION.INPUT_VIDEO != ""
        ), "Must specify a data source as input."

        self.source = (
            cfg.VISUALIZATION.WEBCAM if cfg.VISUALIZATION.WEBCAM > -1 else cfg.VISUALIZATION.INPUT_VIDEO
        )

        self.display_width = cfg.VISUALIZATION.DISPLAY_WIDTH
        self.display_height = cfg.VISUALIZATION.DISPLAY_HEIGHT

        self.cap = cv2.VideoCapture(self.source)

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.id = -1
        self.buffer = []
        self.buffer_size = cfg.VISUALIZATION.BUFFER_SIZE
        self.seq_length = cfg.DATASETS.FRAME_INTERVAL * cfg.DATASETS.NUM_CLIPS

    def __iter__(self):
        return self

    def __next__(self):
        """
        Read and return the required number of frames for 1 clip.
        Returns:
            was_read (bool): False if not enough frames to return.
            task (TaskInfo object): object contains metadata for the current clips.
        """
        self.id += 1
        task = TaskInfo()

        task.img_height = self.display_height
        task.img_width = self.display_width

        frames = []
        if len(self.buffer) != 0:
            frames = self.buffer
        was_read = True
        while was_read and len(frames) < self.seq_length:
            was_read, frame = self.cap.read()
            if not was_read:
                break
            frame = cv2.resize(frame, (self.display_width, self.display_height))
            frames.append(frame)
        if was_read and self.buffer_size != 0:
            self.buffer = frames[-self.buffer_size:]

        task.add_frames(self.id, frames)
        task.num_buffer_frames = 0 if self.id == 0 else self.buffer_size

        return was_read, task

    def clean(self):
        if self.cap:
            self.cap.release()
