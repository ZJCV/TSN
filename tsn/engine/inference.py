# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

import os
import datetime
import torch
import time
from tqdm import tqdm

import tsn.util.logging as logging
from tsn.util.distributed import all_gather, all_reduce, is_master_proc
from tsn.data.build import build_dataloader


@torch.no_grad()
def compute_on_dataset(images, targets, device, model, num_gpus, evaluator):
    images = images.to(device=device, non_blocking=True)
    targets = targets.to(device=device, non_blocking=True)

    outputs = model(images)
    # Gather all the predictions across all the devices to perform ensemble.
    if num_gpus > 1:
        outputs, targets = all_gather([outputs, targets])

    top1, top5 = evaluator.evaluate(outputs, targets, topk=(1, 5), once=False)
    # Gather all the predictions across all the devices.
    if num_gpus > 1:
        top1, top5 = all_reduce([top1, top5])


def inference(cfg, model, device, **kwargs):
    iteration = kwargs.get('iteration', None)
    dataset_name = cfg.DATASETS.TEST.NAME
    num_gpus = cfg.NUM_GPUS

    data_loader = build_dataloader(cfg, is_train=False)
    dataset = data_loader.dataset
    evaluator = data_loader.dataset.evaluator
    evaluator.clean()

    logger = logging.setup_logging(__name__)
    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))
    max_iter = len(data_loader)

    start_training_time = time.time()
    if is_master_proc():
        for iteration, (images, targets) in enumerate(tqdm(data_loader), 0):
            compute_on_dataset(images, targets, device, model, num_gpus, evaluator)
    else:
        for iteration, (images, targets) in enumerate(data_loader, 0):
            compute_on_dataset(images, targets, device, model, num_gpus, evaluator)

    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total evaluate time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    topk_list, cate_topk_dict = evaluator.get()
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

    if is_master_proc():
        output_dir = cfg.OUTPUT.DIR
        if iteration is not None:
            result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
        else:
            result_path = os.path.join(output_dir,
                                       'result_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        with open(result_path, "w") as f:
            f.write(result_str)

    return {'top1': top1_acc, 'top5': top5_acc}


@torch.no_grad()
def do_evaluation(cfg, model, device, **kwargs):
    model.eval()

    return inference(cfg, model, device, **kwargs)
