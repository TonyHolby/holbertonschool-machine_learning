#!/usr/bin/env python3
"""
    A function that normalizes (standardizes) a matrix.
"""
import numpy as np


def normalize(X, m, s):
    """
        Normalizes (standardizes) a matrix.

        Args:
            X (np.ndarray): the numpy.ndarray of shape (d, nx) to normalize :
                - d is the number of data points.
                - nx is the number of features.
            m (np.ndarray): a numpy.ndarray of shape (nx,) that contains the
                mean of all features of X.
            s (np.ndarray): a numpy.ndarray of shape (nx,) that contains the
                standard deviation of all features of X.

        Returns:
            The normalized X matrix.
    """
    nx = X.shape[1]
    X_norm = np.zeros_like(X)

    for feature in range(nx):
        X_norm[:, feature] = (X[:, feature] - m[feature]
                              ) / (s[feature] + 1e-16)

    return X_norm
