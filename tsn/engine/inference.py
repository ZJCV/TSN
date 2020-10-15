# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np

from tsn.util.metrics import topk_accuracy
import tsn.util.logging as logging
from tsn.data.build import build_dataloader


def compute_on_dataset(model, data_loader):
    results_dict = {}
    cate_acc_dict = {}
    acc_top1 = list()
    acc_top5 = list()

    for batch in tqdm(data_loader):
        images, targets = batch
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            outputs = model(images.cuda(non_blocking=True)).to(cpu_device)

            topk_list = topk_accuracy(outputs, targets, topk=(1, 5))
            acc_top1.append(topk_list[0].item())
            acc_top5.append(topk_list[1].item())

            outputs = outputs.numpy()
            preds = np.argmax(outputs, 1)
            targets = targets.numpy()
            for target, pred in zip(targets, preds):
                results_dict.update({
                    str(target):
                        results_dict.get(str(target), 0) + 1
                })
                cate_acc_dict.update({
                    str(target):
                        cate_acc_dict.get(str(target), 0) + int(target == pred)
                })

    return results_dict, cate_acc_dict, acc_top1, acc_top5


def inference(cfg, model, **kwargs):
    iteration = kwargs.get('iteration', None)
    dataset_name = cfg.DATASETS.TEST.NAME
    output_dir = cfg.OUTPUT.DIR

    data_loader = build_dataloader(cfg, is_train=False)
    dataset = data_loader.dataset

    logger = logging.setup_logging()
    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))

    results_dict, cate_acc_dict, acc_top1, acc_top5 = compute_on_dataset(model, data_loader)

    top1_acc = np.mean(acc_top1)
    top5_acc = np.mean(acc_top5)
    result_str = '\ntotal - top_1 acc: {:.3f}, top_5 acc: {:.3f}\n'.format(top1_acc, top5_acc)

    classes = dataset.classes
    for key in sorted(results_dict.keys(), key=lambda x: int(x)):
        total_num = results_dict[key]
        acc_num = cate_acc_dict[key]

        cate_name = classes[int(key)]

        if total_num != 0:
            result_str += '{:<3} - {:<20} - acc: {:.2f}\n'.format(key, cate_name, acc_num / total_num * 100)
        else:
            result_str += '{:<3} - {:<20} - acc: 0.0\n'.format(key, cate_name, acc_num / total_num)
    logger.info(result_str)

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return {'top1': top1_acc, 'top5': top5_acc}


@torch.no_grad()
def do_evaluation(cfg, model, **kwargs):
    model.eval()

    return inference(cfg, model, **kwargs)
