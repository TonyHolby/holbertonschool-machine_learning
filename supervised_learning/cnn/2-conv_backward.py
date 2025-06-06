#!/usr/bin/env python3
"""
    A function that performs back propagation over a convolutional layer of
    a neural network.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
        Performs back propagation over a convolutional layer of a neural
        network.

        Args:
            dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the
            unactivated output of the convolutional layer:
                m is the number of examples.
                h_new is the height of the output.
                w_new is the width of the output.
                c_new is the number of channels in the output.
            A_prev (np.ndarray): a numpy.ndarray of shape
            (m, h_prev, w_prev, c_prev) containing the output
            of the previous layer:
                h_prev is the height of the previous layer.
                w_prev is the width of the previous layer.
                c_prev is the number of channels in the previous layer.
            W (np.ndarray): a numpy.ndarray of shape
            (kh, kw, c_prev, c_new) containing the kernels
            for the convolution:
                kh is the filter height.
                kw is the filter width.
            b (np.ndarray): a numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
            padding (str): a string that is either same or valid, indicating
            the type of padding used.
            stride (tuple): a tuple of (sh, sw) containing the strides
            for the convolution:
                sh is the stride for the height.
                sw is the stride for the width.

        Returns:
            The  partial derivatives with respect to the previous layer
            (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                 (pad_w, pad_w), (0, 0)), mode='constant')
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (pad_h, pad_h),
                                   (pad_w, pad_w), (0, 0)), mode='constant')

    for i in range(m):
        a_prev = A_prev_pad[i]
        da_prev = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev[
                        vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev[vert_start:vert_end,
                            horiz_start:horiz_end,
                            :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i] = da_prev[pad_h:-pad_h or None,
                                 pad_w:-pad_w or None, :]
        else:
            dA_prev[i] = da_prev

    return dA_prev, dW, db
