#!/usr/bin/env python3
"""
    A script that creates a bidirectional cell of an RNN.
"""
import numpy as np


class BidirectionalCell:
    """
        A class that represents a bidirectional cell of an RNN.
    """
    def __init__(self, i, h, o):
        """
            Initializes a bidirectional cell of an RNN.

            Args:
                i (int): the dimensionality of the data.
                h (int): the dimensionality of the hidden state.
                o (int): the dimensionality of the outputs.
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
            Calculates the hidden state in the forward direction for one time
            step.

            Args:
                x_t (np.ndarray): a numpy.ndarray of shape (m, i) that contains
                    the data input for the cell:
                        m is the batch size for the data.
                h_prev (np.ndarray): a numpy.ndarray of shape (m, h) containing
                    the previous hidden state.

            Returns:
                h_next, the next hidden state.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next
