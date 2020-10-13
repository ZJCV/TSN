# -*- coding: utf-8 -*-

"""
@date: 2020/10/9 下午10:50
@file: dense_sample.py
@author: zj
@description: 
"""

import numpy as np

from .seg_sample import SegmentedSample


class DenseSample(SegmentedSample):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
    """

    def __init__(self,
                 clip_len,
                 frame_interval,
                 num_clips,
                 is_train=True,
                 start_index=0):
        super().__init__(clip_len,
                         frame_interval,
                         num_clips,
                         is_train=is_train,
                         start_index=start_index)

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        total_clip_len = self.clip_len * self.frame_interval * self.num_clips
        start_range = num_frames - total_clip_len

        if start_range > 0:
            start_idx = np.random.randint(start_range)
            end_idx = start_idx + total_clip_len
            clip_offsets = np.linspace(start_idx, end_idx, num=self.num_clips)
        elif num_frames > (self.clip_len * self.num_clips):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - self.clip_len * self.num_clips + 1, size=self.num_clips))
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        total_clip_len = self.clip_len * self.frame_interval * self.num_clips
        start_range = num_frames - total_clip_len

        if start_range > 0:
            start_idx = start_range // 2
            end_idx = start_idx + total_clip_len
            clip_offsets = np.linspace(start_idx, end_idx, num=self.num_clips)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets
