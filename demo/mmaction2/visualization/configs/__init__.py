# -*- coding: utf-8 -*-

"""
@date: 2021/1/25 下午4:33
@file: __init__.py.py
@author: zj
@description: 
"""

from tsn.config import _C
from .demo import add_config

add_config(_C)

cfg = _C.clone()
