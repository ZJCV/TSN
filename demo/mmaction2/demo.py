# -*- coding: utf-8 -*-

"""
@date: 2021/1/14 下午2:51
@file: demo.py
@author: zj
@description: 
"""

from collections import deque
from operator import itemgetter
from threading import Thread

import numpy as np
import torch

from tsn.model.recognizers.build import build_recognizer
from tsn.data.transforms.build import build_transform
from tsn.util.distributed import get_device

from .constant import *
from .parser import parse_args, load_config


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    while True:
        msg = 'Waiting for action ...'
        ret, frame = cap.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    # 如果本次检测成绩小于阈值，则不修改显示字符串
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
        elif len(text_info):
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR, THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def inference(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    score_cache = deque()
    scores_sum = 0
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                cur_data = cur_windows[frame_interval // 2::frame_interval]

        images = torch.stack(cur_data).transpose(0, 1).unsqueeze(0).to(device=device, non_blocking=True)
        with torch.no_grad():
            output_dict = model(images)
            scores = torch.softmax(output_dict['probs'], dim=1).cpu().numpy()[0]

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()


def init():
    # 帧队列
    global frame_queue
    # 捕获设备
    global cap
    # 当前帧
    global frame
    # 检测结果
    global results
    # 可以显示结果的最小阈值
    global threshold
    # 采样长度
    global sample_length
    # 帧间隔
    global frame_interval
    #
    global data
    #
    global test_pipeline
    # 检测模型
    global model
    # 设备
    global device
    # 平均前N检测成绩的长度
    global average_size
    # 标签列表
    global label
    # 结果队列
    global result_queue


def main():
    init()

    args = parse_args()
    cfg = load_config(args)

    average_size = args.average_size
    threshold = args.threshold
    device = get_device(args.device)
    cap = cv2.VideoCapture(args.video_path)
    with open(args.label, 'r') as f:
        label = [line.strip().split(' ')[1] for line in f]

    model = build_recognizer(cfg, device)
    # prepare test pipeline from non-camera pipeline
    test_transform = build_transform(cfg, is_train=False)
    sample_length = cfg.DATASETS.CLIP_LEN * cfg.DATASETS.NUM_CLIPS * cfg.DATASETS.FRAME_INTERVAL
    frame_interval = cfg.DATASETS.FRAME_INTERVAL
    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(cfg), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
