#!/usr/bin/env python3
"""
    A function that performs pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        Performs pooling on images.

        Args:
            images (np.ndarray): a numpy.ndarray with shape (m, h, w, c)
            containing multiple images:
                m is the number of images.
                h is the height in pixels of the images.
                w is the width in pixels of the images.
                c is the number of channels in the image.
            kernel_shape (tuple): a tuple of (kh, kw) containing the kernel
            shape for the pooling:
                kh is the height of the kernel.
                kw is the width of the kernel.
            stride (tuple): a tuple of (sh, sw):
                sh is the stride for the height of the image.
                sw is the stride for the width of the image.
            mode indicates the type of pooling:
                max indicates max pooling.
                avg indicates average pooling.

        Returns:
            A numpy.ndarray containing the pooled images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    c = images.shape[3]
    kh, kw = kernel_shape
    sh, sw = stride

    output_height = ((h - kh) // sh) + 1
    output_width = ((w - kw) // sw) + 1

    pooled_image = np.zeros((m, output_height, output_width, c))

    for i in range(output_height):
        for j in range(output_width):
            vertical_start = i * sh
            vertical_end = vertical_start + kh
            horizontal_start = j * sw
            horizontal_end = horizontal_start + kw
            window = images[:, vertical_start:vertical_end,
                            horizontal_start:horizontal_end, :]

            if mode == 'max':
                pooled_image[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled_image[:, i, j, :] = np.mean(window, axis=(1, 2))

    return pooled_image
