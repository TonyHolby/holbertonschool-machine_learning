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
        forward_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(forward_concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
            Calculates the hidden state in the backward direction for one time
            step.

            Args:
                x_t (np.ndarray): a numpy.ndarray of shape (m, i) that contains
                    the data input for the cell:
                        m is the batch size for the data.
                h_next (np.ndarray): a numpy.ndarray of shape (m, h) containing
                    the next hidden state.

            Returns:
                h_prev, the previous hidden state.
        """
        backward_concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(backward_concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
            Calculates all outputs for the RNN.

            Args:
                H (np.ndarray): a numpy.ndarray of shape (t, m, 2 * h) that
                contains the concatenated hidden states from both directions,
                excluding their initialized states:
                    t is the number of time steps.
                    m is the batch size for the data.
                    h is the dimensionality of the hidden states.

            Returns:
                Y, the outputs.
        """
        t = H.shape[0]
        Y = []
        for step in range(t):
            y_t = np.dot(H[step], self.Wy) + self.by
            exp_logits = np.exp(y_t - np.max(y_t, axis=1, keepdims=True))
            softmax_activation = exp_logits / np.sum(
                exp_logits, axis=1, keepdims=True)
            Y.append(softmax_activation)

        Y = np.array(Y)

        return Y
