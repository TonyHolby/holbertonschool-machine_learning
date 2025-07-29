#!/usr/bin/env python3
"""
    A function that calculates the total intra-cluster variance for a data set.
"""
import numpy as np


def variance(X, C):
    """
        Calculates the total intra-cluster variance for a data set.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            C (np.ndarray): a numpy.ndarray of shape (k, d) containing the
            centroid means for each cluster.

        Returns:
            var, or None on failure (var is the total variance).
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None

    if len(X.shape) != 2 or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    euclidean_distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    closest_cluster = np.argmin(euclidean_distances, axis=1)
    var = np.sum(np.linalg.norm(X - C[closest_cluster], axis=1) ** 2)

    return var
