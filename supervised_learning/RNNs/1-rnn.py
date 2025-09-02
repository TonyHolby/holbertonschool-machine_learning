#!/usr/bin/env python3
"""
    A function that performs forward propagation for a simple RNN.
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
        Performs forward propagation for a simple RNN.

        rnn_cell is an instance of RNNCell that will be used for the forward
            propagation.
        X (np.ndarray): the data to be used, given as a numpy.ndarray of shape
            (t, m, i):
                t is the maximum number of time steps.
                m is the batch size.
                i is the dimensionality of the data.
        h_0 (np.ndarray): the initial hidden state, given as a numpy.ndarray of
            shape (m, h):
                h is the dimensionality of the hidden state.

        Returns:
            H, Y:
                H is a numpy.ndarray containing all of the hidden states.
                Y is a numpy.ndarray containing all of the outputs.
    """
    t, m = X.shape[0], X.shape[1]
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        h_prev = H[step]
        x_t = X[step]

        h_next, y = rnn_cell.forward(h_prev, x_t)

        H[step + 1] = h_next
        Y[step] = y

    return H, Y
