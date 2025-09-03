#!/usr/bin/env python3
"""
    A script that creates a gated recurrent unit.
"""
import numpy as np


class GRUCell:
    """
        A class that represents a gated recurrent unit.
    """
    def __init__(self, i, h, o):
        """
            Initializes a GRU unit.

            Args:
                i (int): the dimensionality of the data.
                h (int): the dimensionality of the hidden state.
                o (int): the dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Performs forward propagation for one time step.

            Args:
                x_t (np.ndarray): a numpy.ndarray of shape (m, i) that contains
                    the data input for the cell:
                        m is the batche size for the data.
                h_prev is a numpy.ndarray of shape (m, h) containing the
                    previous hidden state.

            Returns:
                h_next, y:
                    h_next is the next hidden state.
                    y is the output of the cell.
        """
        update_concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(np.dot(update_concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.dot(update_concat, self.Wr) + self.br)))

        reset_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(reset_concat, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde

        logits = np.dot(h_next, self.Wy) + self.by
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
