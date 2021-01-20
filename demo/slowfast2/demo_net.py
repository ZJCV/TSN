#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import tqdm

from .visualizer import AsyncVis, VideoVisualizer
from .predictor import ActionPredictor, AsyncActionPredictor

from .configs import cfg
from .utils.parser import load_config, parse_args
from .manager import VideoManager, ThreadVideoManager

from tsn.util import logging

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncActionPredictor(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
            cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    start = time.time()
    if cfg.DEMO.THREAD_ENABLE:
        frame_provider = ThreadVideoManager(cfg)
    else:
        frame_provider = VideoManager(cfg)

    for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
        frame_provider.display(task)

    frame_provider.join()
    frame_provider.clean()
    logger.info("Finish demo in: {}".format(time.time() - start))


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg)


if __name__ == "__main__":
    main()
