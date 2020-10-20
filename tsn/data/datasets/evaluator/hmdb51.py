# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午11:57
@file: hmdb51.py
@author: zj
@description: 
"""

import torch
import numpy as np

from .base_evaluator import BaseEvaluator
from tsn.util.metrics import topk_accuracy


class HMDB51Evaluator(BaseEvaluator):

    def __init__(self, classes, topk=(1,)):
        super().__init__(classes, topk=topk)

        self._init()

    def _init(self):
        self.topk_list = list()
        self.cate_acc_dict = dict()
        self.cate_num_dict = dict()

    def evaluate(self, outputs, targets, once=False):
        res = topk_accuracy(outputs, targets, topk=self.topk)
        if once:
            topk_dict = dict()
            for acc, name in zip(res, self.topk):
                topk_dict[f'acc_{name}'] = acc
            return topk_dict

        self.topk_list.append(torch.stack(res).cpu().numpy())
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for target, pred in zip(targets.cpu().numpy(), preds):
            self.cate_num_dict.update({
                str(target):
                    self.cate_num_dict.get(str(target), 0) + 1
            })
            self.cate_acc_dict.update({
                str(target):
                    self.cate_acc_dict.get(str(target), 0) + int(target == pred)
            })

    def get(self):
        if len(self.topk_list) == 0:
            return None, None

        topk_list = np.mean(np.array(self.topk_list), axis=0)
        cate_topk_dict = dict()
        for key in self.cate_num_dict.keys():
            total_num = self.cate_num_dict[key]
            acc_num = self.cate_acc_dict[key]
            class_name = self.classes[int(key)]

            cate_topk_dict[class_name] = 1.0 * acc_num / total_num if total_num != 0 else 0.0

        return topk_list, cate_topk_dict

    def clean(self):
        self._init()
