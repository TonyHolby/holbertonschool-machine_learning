#!/usr/bin/env python3
"""
    A function that calculates the expectation step in the EM algorithm for a
    GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
        Calculates the expectation step in the EM algorithm for a GMM.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            pi (np.ndarray): a numpy.ndarray of shape (k,) containing the
            priors for each cluster.
            m (np.ndarray): a numpy.ndarray of shape (k, d) containing the
            centroid means for each cluster.
            S (np.ndarray): a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for each cluster.

        Returns:
            g, l, or None, None on failure:
                g is a numpy.ndarray of shape (k, n) containing the posterior
                probabilities for each data point in each cluster.
                l is the total log likelihood.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))
    for i in range(k):
        pdf_i = pdf(X, m[i], S[i])
        if pdf_i is None:
            return None, None
        g[i] = pi[i] * pdf_i

    total_likelihood = np.sum(g, axis=0)
    g /= total_likelihood
    total_log_likelihood = np.sum(np.log(total_likelihood))

    return g, total_log_likelihood
