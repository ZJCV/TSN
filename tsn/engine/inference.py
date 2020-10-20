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

import tsn.util.logging as logging
from tsn.data.build import build_dataloader


@torch.no_grad()
def compute_on_dataset(model, data_loader, device):
    evaluator = data_loader.dataset.evaluator
    evaluator.clean()
    for images, targets in tqdm(data_loader):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        outputs = model(images)
        evaluator.evaluate(outputs, targets, once=False)

    topk_list, cate_topk_dict = evaluator.get()
    return topk_list, cate_topk_dict


def inference(cfg, model, device, **kwargs):
    iteration = kwargs.get('iteration', None)
    dataset_name = cfg.DATASETS.TEST.NAME
    output_dir = cfg.OUTPUT.DIR

    data_loader = build_dataloader(cfg, is_train=False)
    dataset = data_loader.dataset

    logger = logging.setup_logging()
    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))

    topk_list, cate_topk_dict = compute_on_dataset(model, data_loader, device)

    top1_acc, top5_acc = topk_list
    result_str = '\ntotal - top_1 acc: {:.3f}, top_5 acc: {:.3f}\n'.format(top1_acc, top5_acc)

    classes = dataset.classes
    for idx in range(len(classes)):
        class_name = classes[idx]
        cate_acc = cate_topk_dict[class_name]

        if cate_acc != 0:
            result_str += '{:<3} - {:<20} - acc: {:.2f}\n'.format(idx, class_name, cate_acc * 100)
        else:
            result_str += '{:<3} - {:<20} - acc: 0.0\n'.format(idx, class_name)
    logger.info(result_str)

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return {'top1': top1_acc, 'top5': top5_acc}


@torch.no_grad()
def do_evaluation(cfg, model, device, **kwargs):
    model.eval()

    return inference(cfg, model, device, **kwargs)
