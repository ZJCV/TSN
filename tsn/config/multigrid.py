# -*- coding: utf-8 -*-

"""
@date: 2020/11/19 下午6:52
@file: multigrid.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_visualization_config(_C):
    # ---------------------------------------------------------------------------- #
    # Sampler
    # ---------------------------------------------------------------------------- #
    _C.SAMPLER = CN()

    _C.SAMPLER.MULTIGRID = CN()
    _C.SAMPLER.MULTIGRID.DEFAULT_S = 0
    # Enable short cycles.
    _C.SAMPLER.MULTIGRID.SHORT_CYCLE = False
    # Short cycle additional spatial dimensions relative to the default crop size.
    _C.SAMPLER.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]
