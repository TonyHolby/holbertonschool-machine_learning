#!/usr/bin/env python3
"""
    A function that determines if you should stop gradient descent early.
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        Determines if you should stop gradient descent early.

        Args:
            cost (float): the current validation cost of the neural network.
            opt_cost (float): the lowest recorded validation cost of the
            neural network.
            threshold (float): the threshold used for early stopping.
            patience (int): the patience count used for early stopping.
            count (int): the count of how long the threshold has not been met.

        Returns:
            A boolean of whether the network should be stopped early, followed
            by the updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop = count >= patience

    return stop, count
