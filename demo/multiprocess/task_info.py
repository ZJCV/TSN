# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午3:31
@file: task_info.py
@author: zj
@description: 
"""


class TaskInfo:

    def __init__(self):
        # 当前任务ID
        self.id = -1
        # 帧列表
        self.frames = None
        # 图像宽/高
        self.image_width = 0
        self.image_height = 0
        # 当前缓存帧数
        self.num_buffer_frames = 0
        # 动作预测结果
        self.action_preds = None

    def add_frames(self, idx, frames):
        """
        Add the clip and corresponding id.
        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.id = idx

    def add_action_preds(self, preds):
        """
        Add the corresponding action predictions.
        """
        self.action_preds = preds
