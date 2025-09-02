#!/usr/bin/env python3
"""
    A script that creates a cell of a simple RNN.
"""
import numpy as np


class RNNCell:
    """
        A class named RNNCell that represents a cell of a simple RNN.
    """
    def __init__(self, i, h, o):
        """
            Initializes a cell of the RNN.

            Args:
                i (int): the dimensionality of the data.
                h (int): the dimensionality of the hidden state.
                o (int): the dimensionality of the outputs.
        """
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
                h_prev (np.ndarray): a numpy.ndarray of shape (m, h) containing
                    the previous hidden state.

            Returns:
                h_next, y:
                    h_next is the next hidden state.
                    y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        z = np.dot(h_next, self.Wy) + self.by
        exponential_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exponential_z / np.sum(exponential_z, axis=1, keepdims=True)

        return h_next, y
