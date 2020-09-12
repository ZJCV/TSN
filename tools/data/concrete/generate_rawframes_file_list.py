# -*- coding: utf-8 -*-

"""
@date: 2020/9/9 下午3:13
@file: generate_rawframes_file_list.py
@author: zj
@description: 
"""

import os
import glob
import numpy as np

rawframes_dir = 'data/concrete/rawframes'
res_path = 'data/concrete/concrete_rawframes.txt'

if __name__ == '__main__':
    cate_list = os.listdir(rawframes_dir)
    # print(cate_list)

    file_list = []
    for cate in cate_list:
        cate_dir = os.path.join(rawframes_dir, cate)
        catelog_list = os.listdir(cate_dir)
        # print(catelog_list)

        for catelog in catelog_list:
            catelog_dir = os.path.join(cate_dir, catelog)
            img_list = os.listdir(catelog_dir)
            # print(img_list)

            file_list.append([catelog_dir, len(img_list), cate])
    print(file_list)

    np.savetxt(res_path, file_list, fmt='%s', delimiter=' ')
