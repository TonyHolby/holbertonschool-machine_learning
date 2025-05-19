#!/usr/bin/env python3
"""
    A function that performs a convolution on grayscale images with custom
    padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        Performs a convolution on grayscale images with custom padding.

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
            padding (tuple): a tuple of (ph, pw):
                ph is the padding for the height of the image
                pw is the padding for the width of the image.

        Returns:
            A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding

    padded_images = np.pad(images,
                           pad_width=((0, 0),
                                      (ph, ph),
                                      (pw, pw)
                                      ),
                           mode='constant', constant_values=0
                           )

    output_height = h + 2 * ph - kh + 1
    output_width = w + 2 * pw - kw + 1

    convolved_image = np.zeros((m, output_height, output_width))

    for h_idx in range(h):
        for w_idx in range(w):
            convolved_image[:, h_idx, w_idx] = np.sum(
                padded_images[:, w_idx:w_idx+kh, h_idx:h_idx+kw]
                * kernel, axis=(1, 2)
                )

    return convolved_image
