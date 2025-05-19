#!/usr/bin/env python3
"""
    A function that calculates the cost of a neural network with L2
    regularization.
"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Calculates the cost of a neural network with L2 regularization.

        Args:
            cost (float): the cost of the network without L2
                regularization.
            lambtha (float): the regularization parameter.
            weights (dict): a dictionary of the weights and biases
                (numpy.ndarrays) of the neural network.
            L (int): the number of layers in the neural network.
            m (int): the number of data points used.

        Returns:
            The cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        l2_sum += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * l2_sum
    cost_with_l2 = cost + l2_cost

    return cost_with_l2
