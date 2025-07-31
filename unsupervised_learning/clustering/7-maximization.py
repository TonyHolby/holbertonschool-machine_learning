#!/usr/bin/env python3
"""
    A function that calculates the maximization step in the EM algorithm for a
    GMM.
"""
import numpy as np


def maximization(X, g):
    """
        Calculates the maximization step in the EM algorithm for a GMM.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            g (np.ndarray): a numpy.ndarray of shape (k, n) containing the
            posterior probabilities for each data point in each cluster.

        Returns:
            pi, m, S, or None, None, None on failure:
                pi is a numpy.ndarray of shape (k,) containing the updated
                priors for each cluster.
                m is a numpy.ndarray of shape (k, d) containing the updated
                centroid means for each cluster.
                S is a numpy.ndarray of shape (k, d, d) containing the updated
                covariance matrices for each cluster.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2\
            or g.shape[1] != X.shape[0]\
            or not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    Nk = np.sum(g, axis=1)
    pi = Nk / n
    m = (g @ X) / Nk[:, np.newaxis]
    S = np.zeros((k, d, d))

    for i in range(k):
        x_minus_mean = X - m[i]
        weighted = g[i, :, np.newaxis] * x_minus_mean
        S[i] = (weighted.T @ x_minus_mean) / Nk[i]

    return pi, m, S
