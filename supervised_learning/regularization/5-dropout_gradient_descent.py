#!/usr/bin/env python3
"""
    A function that updates the weights of a neural network with Dropout regularization using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        Updates the weights of a neural network with Dropout regularization using gradient descent.

        Args:
            Y (np.ndarray): a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data:
                classes is the number of classes.
                m is the number of data points.
            weights (dict): a dictionary of the weights and biases of
            the neural network.
            cache (dict): a dictionary of the outputs and dropout masks of each layer of the neural network
            alpha (float): the learning rate.
            keep_prob (float): the probability that a node will be kept.
            L (int): the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_previous = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_previous.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dropout = cache['Dropout' + str(layer - 1)]
            dA_prev = (dA_prev * dropout) / keep_prob

            A_previous = cache['A' + str(layer - 1)]
            dZ = dA_prev * (1 - A_previous ** 2)
