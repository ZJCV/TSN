# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午8:00
@file: trainer.py
@author: zj
@description: 
"""

import os
import datetime
import time
import torch

from tsn.util.metric_logger import MetricLogger, update_stats, log_iter_stats, log_epoch_stats
from tsn.util.distributed import is_master_proc, synchronize
from tsn.util import logging
from tsn.engine.inference import do_evaluation
from tsn.data.build import shuffle_dataset


def do_train(cfg, arguments,
             data_loader, model, criterion, optimizer, lr_scheduler,
             checkpointer, device):
    logger = logging.setup_logging(__name__)
    meters = MetricLogger()
    evaluator = data_loader.dataset.evaluator
    summary_writer = None
    use_tensorboard = cfg.TRAIN.USE_TENSORBOARD
    if is_master_proc() and use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    log_step = cfg.TRAIN.LOG_STEP
    save_epoch = cfg.TRAIN.SAVE_EPOCH
    eval_epoch = cfg.TRAIN.EVAL_EPOCH
    max_epoch = cfg.TRAIN.MAX_EPOCH
    start_epoch = arguments['cur_epoch']
    epoch_iters = len(data_loader)
    max_iter = (max_epoch - start_epoch) * epoch_iters

    synchronize()
    model.train()
    logger.info("Start training ...")
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch))
    start_training_time = time.time()
    end = time.time()
    for cur_epoch in range(start_epoch, max_epoch + 1):
        shuffle_dataset(data_loader, cur_epoch)
        for iteration, (images, targets) in enumerate(data_loader):
            images = images.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            output_dict = model(images)
            loss_dict = criterion(output_dict, targets)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_list = evaluator.evaluate_train(output_dict, targets)
            update_stats(cfg.NUM_GPUS, meters, loss_dict, acc_list)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            if (iteration + 1) % log_step == 0:
                logger.info(log_iter_stats(
                    iteration, epoch_iters, cur_epoch, max_epoch, optimizer.param_groups[0]['lr'], meters))
            if is_master_proc() and summary_writer:
                global_step = (cur_epoch - 1) * epoch_iters + (iteration + 1)
                for name, meter in meters.meters.items():
                    summary_writer.add_scalar('{}/avg'.format(name), float(meter.avg),
                                              global_step=global_step)
                    summary_writer.add_scalar('{}/global_avg'.format(name), meter.global_avg,
                                              global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        logger.info(log_epoch_stats(epoch_iters, cur_epoch, max_epoch, optimizer.param_groups[0]['lr'], meters))
        arguments["cur_epoch"] = cur_epoch
        lr_scheduler.step()
        if is_master_proc() and save_epoch > 0 and cur_epoch % save_epoch == 0 and cur_epoch != max_epoch:
            checkpointer.save("model_{:04d}".format(cur_epoch), **arguments)
        if eval_epoch > 0 and cur_epoch % eval_epoch == 0 and cur_epoch != max_epoch:
            eval_results = do_evaluation(cfg, model, device, cur_epoch=cur_epoch)
            model.train()
            if is_master_proc() and summary_writer:
                for key, value in eval_results.items():
                    summary_writer.add_scalar(f'eval/{key}', value, global_step=cur_epoch + 1)

    if eval_epoch > 0:
        logger.info('Start final evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        eval_results = do_evaluation(cfg, model, device)

        if is_master_proc() and summary_writer:
            for key, value in eval_results.items():
                summary_writer.add_scalar(f'eval/{key}', value, global_step=arguments["cur_epoch"])
            summary_writer.close()
    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
