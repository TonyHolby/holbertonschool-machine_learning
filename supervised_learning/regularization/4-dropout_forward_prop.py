#!/usr/bin/env python3
"""
    A function that conducts forward propagation using Dropout.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using Dropout.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (nx, m)
            containing the input data for the network:
                nx is the number of input features.
                m is the number of data points.
            weights (dict): a dictionary of the weights and biases of
            the neural network.
            L (int): the number of layers of the network.
            keep_prob (float): the probability that a node will be kept.

        Returns:
            A dictionary containing the outputs of each layer and
            the dropout mask used on each layer.
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            dropout_mask = np.random.rand(*A.shape) < keep_prob
            A = A * dropout_mask / keep_prob
            cache['Dropout' + str(layer)] = dropout_mask

        cache['A' + str(layer)] = A

    return cache
