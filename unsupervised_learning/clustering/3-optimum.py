#!/usr/bin/env python3
"""
    A function that tests for the optimum number of clusters by variance.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
        Tests for the optimum number of clusters by variance.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            kmin (int): a positive integer containing the minimum number of
            clusters to check for (inclusive).
            kmax (int): a positive integer containing the maximum number of
            clusters to check for (inclusive).
            iterations (int): a positive integer containing the maximum number
            of iterations for K-means.

        Returns:
            results, d_vars, or None, None on failure:
                results is a list containing the outputs of K-means for each
                cluster size.
                d_vars is a list containing the difference in variance from
                the smallest cluster size for each cluster size.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax < 1 or kmax <= kmin:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for cluster in range(kmin, kmax + 1):
        C, clss = kmeans(X, cluster, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        var = variance(X, C)
        if var is None or var == 0.0:
            return None, None

        d_vars.append(var)

    reference_var = d_vars[0]
    d_vars = [(reference_var - value) for value in d_vars]

    return results, d_vars
