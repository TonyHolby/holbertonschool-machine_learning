#!/usr/bin/env python3
"""
    A function that performs forward and backward propagations for a
    bidirectional RNN and computes the output from concatenated hidden states.
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
        Performs forward and backward propagations for a bidirectional RNN
        and computes the output from concatenated hidden states.

        Args:
            bi_cell: an instance of BidirectinalCell that will be used for the
                forward propagation.
            X (np.ndarray): the data to be used, given as a numpy.ndarray
                of shape (t, m, i):
                    t is the maximum number of time steps.
                    m is the batch size.
                    i is the dimensionality of the data.
            h_0 (np.ndarray): the initial hidden state in the forward
                direction, given as a numpy.ndarray of shape (m, h):
                    h is the dimensionality of the hidden state.
            h_t (np.ndarray): the initial hidden state in the backward
                direction, given as a numpy.ndarray of shape (m, h).

        Returns:
            H, Y:
                H is a numpy.ndarray containing all of the concatenated hidden
                states.
                Y is a numpy.ndarray containing all of the outputs.
    """
    t, m = X.shape[0], X.shape[1]
    h = h_0.shape[1]

    H_forward = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    H_backward = np.zeros((t, m, h))
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)

    return H, Y
