# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午4:37
@file: three_crop.py
@author: zj
@description: 
"""

import numbers
from opencv_transforms import transforms
from opencv_transforms import functional as F


class ThreeCrop(object):
    """Crop the given PIL Image into left/center/right or up/center/down

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return three_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def three_crop(img, size):
    """Crop the given Numpy ndArray Image into left/center/right or up/center/down.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    assert isinstance(img, np.ndarray)
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_height, image_width = img.shape[:2]
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    center = F.center_crop(img, (crop_height, crop_width))
    if image_height > image_width:
        # crop up/center/down
        left = int(round((image_width - crop_width) / 2.))
        crop_top = F.crop(img, 0, left, crop_height, crop_width)

        top = int(round(image_height - crop_height))
        crop_down = F.crop(img, top, left, crop_height, crop_width)
        res = (crop_top, center, crop_down)
    else:
        # crop left/center/right
        top = int(round(image_height - crop_height) / 2.)
        crop_left = F.crop(img, top, 0, crop_height, crop_width)

        left = int(round(image_width - crop_width))
        crop_right = F.crop(img, top, left, crop_height, image_width)
        res = (crop_left, center, crop_right)

    return res


if __name__ == '__main__':
    model = ThreeCrop((255, 255))

    import numpy as np

    # left/center/right
    img = np.arange(255 * 320 * 3).reshape(255, 320, 3).astype(np.uint8)
    crops = model(img)
    print(len(crops))
    for crop in crops:
        print(crop.shape)

    # up/center/down
    img = np.arange(255 * 320 * 3).reshape(320, 255, 3).astype(np.uint8)
    crops = model(img)
    print(len(crops))
    for crop in crops:
        print(crop.shape)
