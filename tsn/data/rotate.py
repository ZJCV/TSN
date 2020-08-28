# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 上午11:14
@file: rotate.py
@author: zj
@description: 
"""

import math
import random
import cv2
import numpy as np


class Rotate:

    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)

        angle = random.randint(0, 359)
        rotate_img = rotate(img, angle)

        return rotate_img, angle


def rotate(img, degree):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    dst_h = int(w * math.fabs(math.sin(math.radians(degree))) + h * math.fabs(math.cos(math.radians(degree))))
    dst_w = int(h * math.fabs(math.sin(math.radians(degree))) + w * math.fabs(math.cos(math.radians(degree))))

    matrix = cv2.getRotationMatrix2D(center, degree, 1)
    matrix[0, 2] += dst_w // 2 - center[0]
    matrix[1, 2] += dst_h // 2 - center[1]
    dst_img = cv2.warpAffine(img, matrix, (dst_w, dst_h), borderValue=(255, 255, 255))

    # imshow(img, 'src')
    # imshow(dst_img, 'dst')
    # cv2.waitKey(0)
    return dst_img
