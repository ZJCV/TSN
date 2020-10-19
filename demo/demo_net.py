#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import tqdm

from tsn.visualization.manager import VideoManager, ThreadVideoManager
from tsn.visualization.visalizer import AsyncVis
from tsn.visualization.predictor import ActionPredictor, AsyncActionPredictor
from tsn.util.parser import parse_train_args, load_config

from tsn.util import logging

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            tsn/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT.DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)

    async_vis = AsyncVis(cfg, n_workers=cfg.VISUALIZATION.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncActionPredictor(cfg=cfg, async_vis=async_vis)

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
            tsn/config/defaults.py
    """
    start = time.time()
    if cfg.VISUALIZATION.THREAD_ENABLE:
        frame_provider = ThreadVideoManager(cfg)
    else:
        frame_provider = VideoManager(cfg)

    for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
        # for task in run_demo(cfg, frame_provider):
        frame_provider.display(task)

    frame_provider.join()
    frame_provider.clean()
    logger.info("Finish demo in: {}".format(time.time() - start))


def main():
    """
    Main function to spawn the train and test predict.
    """
    args = parse_train_args()
    cfg = load_config(args)

    # Run demo.
    if cfg.VISUALIZATION.ENABLE:
        demo(cfg)


if __name__ == "__main__":
    main()
