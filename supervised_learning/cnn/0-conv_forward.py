#!/usr/bin/env python3
"""
    A function that performs forward propagation over a convolutional layer
    of a neural network.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        Performs forward propagation over a convolutional layer of a neural
        network.

        Args:
            A_prev (np.ndarray): a numpy.ndarray of shape
            (m, h_prev, w_prev, c_prev) containing the output
            of the previous layer:
                m is the number of examples.
                h_prev is the height of the previous layer.
                w_prev is the width of the previous layer.
                c_prev is the number of channels in the previous layer.
            W (np.ndarray): a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution:
                kh is the filter height.
                kw is the filter width.
                c_prev is the number of channels in the previous layer.
                c_new is the number of channels in the output.
            b (np.ndarray): a numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
            activation (str): an activation function applied to the
            convolution.
            padding (str): a string that is either same or valid,
            indicating the type of padding used.
            stride (tuple): a tuple of (sh, sw) containing the strides
            for the convolution:
                sh is the stride for the height.
                sw is the stride for the width.

        Returns:
            The output of the convolutional layer.
    """
    m, h_prev, w_prev = A_prev.shape[0], A_prev.shape[1], A_prev.shape[2]
    kh, kw, _, c_new = W.shape[0], W.shape[1], W.shape[2], W.shape[3]
    sh, sw = stride

    if padding == 'same':
        pad_h = max((int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))), 0)
        pad_w = max((int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))), 0)
    else:
        pad_h = pad_w = 0

    output_h = int((h_prev + 2 * pad_h - kh) / sh) + 1
    output_w = int((w_prev + 2 * pad_w - kw) / sw) + 1

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )

    Z = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                vertical_start = i * sh
                vertical_end = vertical_start + kh
                horizontal_start = j * sw
                horizontal_end = horizontal_start + kw

                A_slice = A_prev_padded[:, vertical_start:vertical_end,
                                        horizontal_start:horizontal_end, :]
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k], axis=(1, 2, 3))

    Z += b
    conv_layer_output = activation(Z)

    return conv_layer_output
