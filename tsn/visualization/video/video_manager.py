# -*- coding: utf-8 -*-

"""
@date: 2020/10/13 下午3:04
@file: video_manager.py
@author: zj
@description: 
"""

import cv2

from tsn.util import logging
from tsn.visualization.task_info import TaskInfo

logger = logging.get_logger(__name__)


class VideoManager:
    """
    VideoManager object for getting frames from video source for inference.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        """
        assert (
                cfg.DEMO.WEBCAM > -1 or cfg.DEMO.INPUT_VIDEO != ""
        ), "Must specify a data source as input."

        self.source = (
            cfg.DEMO.WEBCAM if cfg.DEMO.WEBCAM > -1 else cfg.DEMO.INPUT_VIDEO
        )

        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT

        self.cap = cv2.VideoCapture(self.source)

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.output_file = None
        if cfg.DEMO.OUTPUT_FPS == -1:
            self.output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.output_fps = cfg.DEMO.OUTPUT_FPS
        if cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(
                cfg.DEMO.OUTPUT_FILE, fps=self.output_fps
            )
        self.id = -1
        self.buffer = []
        # self.buffer_size = cfg.DEMO.BUFFER_SIZE
        self.seq_length = cfg.DATASETS.CLIP_LEN * cfg.DATASETS.NUM_CLIPS
        self.buffer_size = self.seq_length // 2
        self.test_crop_size = cfg.TRANSFORM.TEST_CROP_SIZE

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
            frames.append(frame)
        if was_read and self.buffer_size != 0:
            self.buffer = frames[-self.buffer_size:]

        task.add_frames(self.id, frames)
        task.num_buffer_frames = 0 if self.id == 0 else self.buffer_size

        return was_read, task

    def get_output_file(self, path, fps=30):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def display(self, task):
        """
        Either display a single frame (BGR image) to a window or write to
        an output file if output path is provided.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for prediction visualization. (e.g. visualized frames.)
        """
        for frame in task.frames[task.num_buffer_frames:]:
            if self.output_file is None:
                cv2.imshow("SlowFast", frame)
                # time.sleep(1 / self.output_fps)
                cv2.waitKey(int(1 / self.output_fps * 1000))
            else:
                self.output_file.write(frame)

    def clean(self):
        """
        Clean up open video files and windows.
        """
        self.cap.release()
        if self.output_file is None:
            cv2.destroyAllWindows()
        else:
            self.output_file.release()

    def start(self):
        return self

    def join(self):
        pass
