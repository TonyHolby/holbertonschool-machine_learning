#!/usr/bin/env python3
""" A function that initializes variables for a Gaussian Mixture Model. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
        Initializes variables for a Gaussian Mixture Model.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            k (int): a positive integer containing the number of clusters.

        Returns:
            pi, m, S, or None, None, None on failure:
                pi is a numpy.ndarray of shape (k,) containing the priors for
                each cluster, initialized evenly.
                m is a numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster, initialized with K-means.
                S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for each cluster, initialized as
                identity matrices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    pi = np.full(shape=(k,), fill_value=(1 / k))
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
