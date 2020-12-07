# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午3:41
@file: video_manager.py
@author: zj
@description: 
"""

import cv2

from .live import Live


class VideoManager:

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

        self.output_file = None
        if cfg.VISUALIZATION.OUTPUT_FPS == -1:
            self.output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.output_fps = cfg.VISUALIZATION.OUTPUT_FPS
        if cfg.VISUALIZATION.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(
                cfg.VISUALIZATION.OUTPUT_FILE, fps=self.output_fps
            )

        self.cap.release()
        self.cap = None

        self.id = -1
        self.buffer = []
        self.buffer_size = cfg.VISUALIZATION.BUFFER_SIZE
        self.seq_length = cfg.DATASETS.FRAME_INTERVAL * cfg.DATASETS.NUM_CLIPS
        self.live = self.init_live()

    def __call__(self, task):
        self.display(task)

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

    def init_live(self):
        return None

        """初始化推流"""
        live = Live(enable=True, way='rtsp', url='rtsp://localhost:554/zj_test',
                    size=(self.display_width, self.display_height), fps=self.output_fps)
        live.run()

        return live

    def display(self, task):
        """
        Either display a single frame (BGR image) to a window or write to
        an output file if output path is provided.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for prediction visualization. (e.g. visualized frames.)
        """
        for frame in task.frames[task.num_buffer_frames:]:
            if self.live is not None:
                self.live.read_frame(frame)

            if self.output_file is None:
                cv2.imshow("SlowFast", frame)
                cv2.waitKey(20)
                # time.sleep(1 / self.output_fps)
            else:
                self.output_file.write(frame)

    def clean(self):
        """
        Clean up open video files and windows.
        """
        if self.cap:
            self.cap.release()
        if self.output_file is None:
            cv2.destroyAllWindows()
        else:
            self.output_file.release()
