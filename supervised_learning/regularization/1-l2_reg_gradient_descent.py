#!/usr/bin/env python3
"""
    A function that updates the weights and biases of a neural network
    using gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Updates the weights and biases of a neural network using
        gradient descent with L2 regularization.

        Args:
            Y (np.ndarray): a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data:
                classes is the number of classes.
                m is the number of data points.
            weights (dict): a dictionary of the weights and biases of the
            neural network.
            cache (dict): a dictionary of the outputs of each layer of the
            neural network.
            alpha (float): the learning rate.
            lambtha (float): the L2 regularization parameter
            L (int): the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_previous = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_previous.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_previous ** 2)
