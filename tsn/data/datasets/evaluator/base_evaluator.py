# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午10:03
@file: base_evaluator.py
@author: zj
@description: 
"""

from abc import ABCMeta, abstractmethod


class BaseEvaluator(metaclass=ABCMeta):

    def __init__(self, classes):
        self.classes = classes

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def clean(self):
        pass
