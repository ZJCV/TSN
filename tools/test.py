# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:15
@file: test.py
@author: zj
@description: 
"""

from tsn.model.build import build_model
from tsn.engine.inference import do_evaluation
from tsn.util.collect_env import collect_env_info
from tsn.util import logging
from tsn.util.distributed import get_device
from tsn.util.parser import parse_test_args, load_test_config
from tsn.util.misc import launch_job
from tsn.util.distributed import setup, cleanup, synchronize


def test(gpu_id, cfg):
    rank = cfg.RANK * cfg.NUM_GPUS + gpu_id
    world_size = cfg.WORLD_SIZE
    setup(rank, world_size, seed=cfg.RNG_SEED)

    model = build_model(cfg, gpu_id=gpu_id)
    device = get_device()

    synchronize()
    do_evaluation(cfg, model, device)

    cleanup()


def main():
    args = parse_test_args()
    cfg = load_test_config(args)

    logger = logging.setup_logging(__name__, output_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg, test)


if __name__ == '__main__':
    main()
