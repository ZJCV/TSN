# -*- coding: utf-8 -*-

"""
@date: 2020/10/13 下午3:10
@file: async_demo.py
@author: zj
@description: 
"""

import queue

import tsn.util.logging as logging
from .async_action_predictor import AsycnActionPredictor

logger = logging.get_logger(__name__)


class AsyncDemo:
    """
    Asynchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
        """
        self.model = AsycnActionPredictor(
            cfg=cfg, result_queue=async_vis.task_queue
        )
        self.async_vis = async_vis

    def put(self, task):
        """
        Put task into task queue for prediction and visualization.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        """
        self.async_vis.get_indices_ls.append(task.id)
        self.model.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task
