#!/usr/bin/env python3
"""
    A function that performs forward propagation for a deep RNN.
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
        Performs forward propagation for a deep RNN.

        Args:
            rnn_cells (list): a list of RNNCell instances of length l that will
                be used for the forward propagation:
                    l is the number of layers.
            X (np.ndarray): the data to be used, given as a numpy.ndarray of
                shape (t, m, i):
                    t is the maximum number of time steps.
                    m is the batch size.
                    i is the dimensionality of the data.
            h_0 (np.ndarray): the initial hidden state, given as a
                numpy.ndarray of shape (l, m, h):
                    h is the dimensionality of the hidden state.

        Returns:
            H, Y:
                H is a numpy.ndarray containing all of the hidden states
                Y is a numpy.ndarray containing all of the outputs.
    """
    t, m = X.shape[0], X.shape[1]
    l, h = h_0.shape[0], h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next
        Y[step] = y

    return H, Y
