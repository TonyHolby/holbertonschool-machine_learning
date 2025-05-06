#!/usr/bin/env python3
"""
    A function that shuffles the data points in two matrices the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
        Shuffles the data points in two matrices the same way.

        Args:
            X (np.ndarray): the first numpy.ndarray of shape (m, nx) to
                shuffle :
                    - m is the number of data points.
                    - nx is the number of features.
            Y (np.ndarray): the second numpy.ndarray of shape (m, ny) to
                shuffle :
                    - m is the same number of data points as in X.
                    - ny is the number of features in Y.

        Returns:
            The shuffled X and Y matrices.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y
