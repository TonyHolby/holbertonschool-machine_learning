#!/usr/bin/env python3
"""
    A function that performs forward propagation over a pooling layer of
    a neural network.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs forward propagation over a pooling layer of a neural
        network.

        Args:
            A_prev (np.ndarray): a numpy.ndarray of shape
            (m, h_prev, w_prev, c_prev) containing the output
            of the previous layer:
                m is the number of examples.
                h_prev is the height of the previous layer.
                w_prev is the width of the previous layer.
                c_prev is the number of channels in the previous layer.
            kernel_shape (tuple): a tuple of (kh, kw) containing the size of
            the kernel for the pooling:
                kh is the kernel height.
                kw is the kernel width.
            stride (tuple): a tuple of (sh, sw) containing the strides
            for the convolution:
                sh is the stride for the height.
                sw is the stride for the width.
            mode (str): a string containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively.

        Returns:
            The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1

    pool_layer_output = np.zeros((m, output_h, output_w, c_prev))

    for i in range(output_h):
        for j in range(output_w):
            vertical_start = i * sh
            vertical_end = vertical_start + kh
            horizontal_start = j * sw
            horizontal_end = horizontal_start + kw

            pool_region = A_prev[:, vertical_start:vertical_end,
                                 horizontal_start:horizontal_end, :]

            if mode == 'max':
                pool_layer_output[:, i, j, :] = np.max(
                    pool_region, axis=(1, 2))
            elif mode == 'avg':
                pool_layer_output[:, i, j, :] = np.mean(
                    pool_region, axis=(1, 2))

    return pool_layer_output
