#!/usr/bin/env python3
"""
    A function that performs a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        Performs a convolution on images with channels.

        Args:
            images (np.ndarray): a numpy.ndarray with shape (m, h, w, c)
            containing multiple images:
                m is the number of images.
                h is the height in pixels of the images.
                w is the width in pixels of the images.
                c is the number of channels in the image.
            kernel (np.ndarray): a numpy.ndarray with shape (kh, kw, c)
            containing the kernel for the convolution:
                kh is the height of the kernel.
                kw is the width of the kernel.
            padding (tuple): either a tuple of (ph, pw), 'same', or 'valid':
                if 'same', performs a same convolution.
                if 'valid', performs a valid convolution.
                if a tuple:
                    ph is the padding for the height of the image.
                    pw is the padding for the width of the image.
            stride (tuple): a tuple of (sh, sw):
                sh is the stride for the height of the image.
                sw is the stride for the width of the image.

        Returns:
            A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + ((h - 1) * sh + kh - h) % 2
        pw = ((w - 1) * sw + kw - w) // 2 + ((w - 1) * sw + kw - w) % 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           pad_width=((0, 0),
                                      (ph, ph),
                                      (pw, pw),
                                      (0, 0)
                                      ),
                           mode='constant',
                           constant_values=0
                           )

    output_height = ((h + 2 * ph - kh) // sh) + 1
    output_width = ((w + 2 * pw - kw) // sw) + 1

    convolved_image = np.zeros((m, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            h_start = i * sh
            w_start = j * sw
            h_end = h_start + kh
            w_end = w_start + kw

            region = padded_images[:, h_start:h_end, w_start:w_end, :]
            convolved_image[:, i, j] = np.sum(
                region * kernel[None, :, :, :], axis=(1, 2, 3))

    return convolved_image
