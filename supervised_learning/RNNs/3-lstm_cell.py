#!/usr/bin/env python3
"""
    A script that implements an LSTM unit.
"""
import numpy as np


class LSTMCell:
    """
        A class that represents an LSTM unit.
    """
    def __init__(self, i, h, o):
        """
            Initializes an LSRM unit.

            Args:
                i (int): the dimensionality of the data.
                h (int): the dimensionality of the hidden state.
                o (int): the dimensionality of the outputs.
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
            Performs forward propagation for one time step.

            Args:
                x_t (np.ndarray): a numpy.ndarray of shape (m, i) that contains
                    the data input for the cell:
                        m is the batche size for the data.
                h_prev (np.ndarray): a numpy.ndarray of shape (m, h) containing
                    the previous hidden state.
                c_prev (np.ndarray): a numpy.ndarray of shape (m, h) containing
                    the previous cell state.

            Returns:
                h_next, c_next, y:
                    h_next is the next hidden state.
                    c_next is the next cell state.
                    y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wf) + self.bf)))
        u_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wu) + self.bu)))
        o_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wo) + self.bo)))

        c_hat = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_hat
        h_next = o_t * np.tanh(c_next)

        logits = np.dot(h_next, self.Wy) + self.by
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, c_next, y
