# -*- coding: utf-8 -*-

"""
@date: 2021/1/14 下午2:51
@file: visualization.py
@author: zj
@description: 
"""

from operator import itemgetter
import numpy as np
import torch
from collections import deque
from threading import Thread

from tsn.model.recognizers.build import build_recognizer
from tsn.data.transforms.build import build_transform
from tsn.util.distributed import get_device, get_local_rank

from demo.mmaction2.visualization.configs.constant import *
from demo.mmaction2.visualization.utils.parser import parse_args, load_config
from demo.mmaction2.visualization.utils.misc import get_cap, get_output_file, get_label


def show_results():
    if output_file is None:
        print('Press "Esc", "q" or "Q" to exit')
    if cap is None:
        return

    text_info = {}
    while True:
        msg = 'Waiting for action ...'
        ret, frame = cap.read()
        if not ret:
            break
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

        if output_file is not None:
            output_file.write(frame)
        else:
            cv2.imshow('camera', frame)
            ch = cv2.waitKey(int(1 / output_fps * 1000))

            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    if cap is not None:
        cap.release()
    if output_file is not None:
        output_file.release()
    cv2.destroyAllWindows()


def inference():
    score_cache = deque()
    scores_sum = 0
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                cur_windows = cur_windows[frame_interval // 2::frame_interval]

        frames = [test_pipeline(frame) for frame in cur_windows]
        images = torch.stack(frames).transpose(0, 1).unsqueeze(0).to(device=device, non_blocking=True)
        with torch.no_grad():
            outputs = model(images)['probs'].cpu().detach()
            scores = torch.softmax(outputs, dim=1).numpy()[0]

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


def main():
    # 帧队列 检测结果 结果队列
    global frame_queue, results, result_queue
    # 捕获设备 输出文件 输出帧率
    global cap, output_file, output_fps
    # 可以显示结果的最小阈值 采样长度 帧间隔 测试图像转换 检测模型 设备 平均前N检测成绩的长度 标签列表
    global threshold, sample_length, frame_interval, test_pipeline, model, device, average_size, label

    args = parse_args()
    cfg = load_config(args)

    cap = get_cap(cfg)
    output_file, output_fps = get_output_file(cfg, cap)
    label = get_label(cfg)

    average_size = cfg.DEMO.AVG_SIZE
    threshold = cfg.DEMO.THRESHOLD

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = get_device(get_local_rank())
    model = build_recognizer(cfg, device)
    # prepare test pipeline from non-camera pipeline
    test_pipeline = build_transform(cfg, is_train=False)

    sample_length = cfg.DATASETS.CLIP_LEN * cfg.DATASETS.NUM_CLIPS * cfg.DATASETS.FRAME_INTERVAL
    frame_interval = cfg.DATASETS.FRAME_INTERVAL
    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
