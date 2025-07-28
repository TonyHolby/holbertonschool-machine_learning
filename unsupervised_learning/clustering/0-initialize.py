#!/usr/bin/env python3
""" A function that initializes cluster centroids for K-means. """
import numpy as np


def initialize(X, k):
    """
        Initializes cluster centroids for K-means.

        Args:
            X is (np.ndarray): numpy.ndarray of shape (n, d) containing the
            dataset that will be used for K-means clustering:
                n is the number of data points.
                d is the number of dimensions for each data point.
            k (int): a positive integer containing the number of clusters.

        Returns:
            A numpy.ndarray of shape (k, d) containing the initialized
            centroids for each cluster, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    return np.random.uniform(mins, maxs, (k, X.shape[1]))
