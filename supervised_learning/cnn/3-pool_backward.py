#!/usr/bin/env python3
"""
    A function that performs back propagation over a pooling layer of a neural
    network.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs back propagation over a pooling layer of a neural network.

        Args:
            dA (np.ndarray): a numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the
            output of the pooling layer:
                m is the number of examples.
                h_new is the height of the output.
                w_new is the width of the output.
                c is the number of channels.
            A_prev (np.ndarray): a numpy.ndarray of shape
            (m, h_prev, w_prev, c) containing the output of the previous layer:
                h_prev is the height of the previous layer.
                w_prev is the width of the previous layer.
            kernel_shape (tuple): a tuple of (kh, kw) containing the size of
            the kernel for the pooling:
                kh is the kernel height.
                kw is the kernel width.
            stride (tuple): a tuple of (sh, sw) containing the strides for
            the pooling:
                sh is the stride for the height.
                sw is the stride for the width.
            mode (str): a string containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively.

        Returns:
            The partial derivatives with respect to the previous layer
            (dA_prev).
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, ch]

                    if mode == 'max':
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                ch] += mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        average = da / (kh * kw)
                        shape = (kh, kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                ch] += np.ones(shape) * average

    return dA_prev
