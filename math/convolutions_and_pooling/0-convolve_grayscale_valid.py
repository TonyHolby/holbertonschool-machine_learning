#!/usr/bin/env python3
"""
    A function that performs a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        Performs a valid convolution on grayscale images.

        Args:
            images (np.ndarray): a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images:
                m is the number of images.
                h is the height in pixels of the images.
                w is the width in pixels of the images.
            kernel (np.ndarray): a numpy.ndarray with shape (kh, kw) containing
            the kernel for the convolution:
                kh is the height of the kernel.
                kw is the width of the kernel.

        Returns:
            A numpy.ndarray containing the convolved images.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved_image = np.zeros((m, output_h, output_w))

    for height_idx in range(output_h):
        for width_idx in range(output_w):
            convolved_image[:, height_idx, width_idx] = np.sum(
                images[:, height_idx:height_idx+kh, width_idx:width_idx+kw]
                * kernel, axis=(1, 2))

    return convolved_image
