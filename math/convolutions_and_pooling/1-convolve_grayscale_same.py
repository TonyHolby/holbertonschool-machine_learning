#!/usr/bin/env python3
"""
    A function that performs a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        Pperforms a same convolution on grayscale images.

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

    pad_h = kh // 2
    pad_w = kw // 2
    pad_top = pad_h
    pad_bottom = pad_h if kh % 2 != 0 else pad_h - 1
    pad_left = pad_w
    pad_right = pad_w if kw % 2 != 0 else pad_w - 1

    padded_images = np.pad(images,
                           pad_width=((0, 0),
                                      (pad_top, pad_bottom),
                                      (pad_left, pad_right)
                                      ),
                           mode='constant', constant_values=0
                           )

    convolved_image = np.zeros((m, h, w))

    for h_idx in range(h):
        for w_idx in range(w):
            convolved_image[:, h_idx, w_idx] = np.sum(
                padded_images[:, h_idx:h_idx+kh, w_idx:w_idx+kw]
                * kernel, axis=(1, 2))

    return convolved_image
