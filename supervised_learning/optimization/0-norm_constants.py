#!/usr/bin/env python3
"""
    A function that calculates the normalization (standardization)
    constants of a matrix.
"""
import numpy as np


def normalization_constants(X):
    """
        Calculates the normalization (standardization) constants of a matrix.

        Args:
            X (np.ndarray): the numpy.ndarray of shape (m, nx) to normalize :
                - m is the number of data points.
                - nx is the number of features.

        Returns:
            The mean and standard deviation of each feature, respectively.
    """
    nx = X.shape[1]
    means = np.zeros(nx)
    standard_deviation = np.zeros(nx)

    for feature in range(nx):
        number_of_features = X[:, feature]
        means[feature] = np.mean(number_of_features)
        standard_deviation[feature] = np.std(number_of_features)

    return means, standard_deviation
