# -*- coding: utf-8 -*-

"""
@date: 2020/10/30 下午3:42
@file: demo.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import time

from tsn.util.parser import load_test_config, parse_test_args
from tsn.multiprocess.stop_token import _StopToken
from tsn.multiprocess.manager.video_provider import VideoProvider
from tsn.multiprocess.manager.video_manager import VideoManager
from tsn.multiprocess.predictor.action_predictor import ActionPredictor
from tsn.multiprocess.visualizer.video_visualizor import VideoVisualizer

time_decay = 0.001


def read(cfg, task_queue):
    provider = VideoProvider(cfg)

    for able_to_read, task in provider:
        if not able_to_read:
            task_queue.put(_StopToken())
            break
        start = time.time()
        task_queue.put(task, block=False)
        print('one put task_queue need: {}'.format(time.time() - start))
    provider.clean()
    time.sleep(100)


def write(cfg, result_queue):
    manager = VideoManager(cfg)

    while True:
        start = time.time()
        if result_queue.empty():
            time.sleep(time_decay)
            continue
        task = result_queue.get()
        end = time.time()
        print('one get result_queue need: {}'.format(end - start))
        if isinstance(task, _StopToken):
            break

        ret = manager(task)
        print('one compute manager need: {}'.format(time.time() - end))
    manager.clean()


def predict(cfg, task_queue, predict_queue):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    predictor = ActionPredictor(cfg)
    while True:
        start = time.time()
        if task_queue.empty():
            time.sleep(time_decay)
            continue
        task = task_queue.get()
        end = time.time()
        print('one get task_queue need: {}'.format(end - start))
        if isinstance(task, _StopToken):
            predict_queue.put(_StopToken())
            break

        task = predictor(task)
        end1 = time.time()
        print('one task predict need: {}'.format(end1 - end))
        predict_queue.put(task, block=False)
        print('one put predict_queue need: {}'.format(time.time() - end1))
    time.sleep(100)


def visualize(cfg, predict_queue, result_queue):
    visualizer = VideoVisualizer(cfg)

    while True:
        start = time.time()
        if predict_queue.empty():
            time.sleep(time_decay)
            continue
        task = predict_queue.get()
        end = time.time()
        print('one get predict_queue need: {}'.format(end - start))
        if isinstance(task, _StopToken):
            result_queue.put(_StopToken())
            break

        task = visualizer(task)
        end1 = time.time()
        print('one compute visualizer need: {}'.format(end1 - end))
        result_queue.put(task, block=False)
        print('one put result_queue need: {}'.format(time.time() - end1))
    time.sleep(100)


def main():
    args = parse_test_args()
    cfg = load_test_config(args)

    # 任务队列，保存待预测数据
    task_queue = mp.Queue()
    # 预测队列，保存待绘制数据
    predict_queue = mp.Queue()
    # 结果队列，保存待显示数据
    result_queue = mp.Queue()

    process_read = mp.Process(target=read, args=(cfg, task_queue), daemon=True)
    process_predict = mp.Process(target=predict, args=(cfg, task_queue, predict_queue), daemon=True)
    process_visualize = mp.Process(target=visualize, args=(cfg, predict_queue, result_queue), daemon=True)
    process_write = mp.Process(target=write, args=(cfg, result_queue))

    process_write.start()
    process_visualize.start()
    process_predict.start()
    time.sleep(2)
    process_read.start()

    process_read.join()
    process_predict.join()
    process_visualize.join()
    process_write.join()


if __name__ == '__main__':
    main()
