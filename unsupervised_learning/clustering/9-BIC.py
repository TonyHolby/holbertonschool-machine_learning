#!/usr/bin/env python3
"""
    A function that finds the best number of clusters for a GMM using the
    Bayesian Information Criterion.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
        Finds the best number of clusters for a GMM using the Bayesian
        Information Criterion.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            kmin (int): a positive integer containing the minimum number of
            clusters to check for (inclusive).
            kmax (int): a positive integer containing the maximum number of
            clusters to check for (inclusive).
            iterations (int): a positive integer containing the maximum number
            of iterations for the EM algorithm.
            tol (float): a non-negative float containing the tolerance for the
            EM algorithm.
            verbose (bool): a boolean that determines if the EM algorithm
            should print information to the standard output.

        Returns:
            best_k, best_result, l_values, b, or None, None, None, None on
            failure:
                best_k is the best value for k based on its BIC.
                best_result is tuple containing pi, m, S:
                    pi is a numpy.ndarray of shape (k,) containing the cluster
                    priors for the best number of clusters.
                    m is a numpy.ndarray of shape (k, d) containing the
                    centroid means for the best number of clusters.
                    S is a numpy.ndarray of shape (k, d, d) containing the
                    covariance matrices for the best number of clusters.
                l_values is a numpy.ndarray of shape (kmax - kmin + 1)
                containing the log likelihood for each cluster size tested.
                b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
                BIC value for each cluster size tested.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is not None and (
        not isinstance(kmax, int) or kmax < 1 or kmax <= kmin):
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    ks = range(kmin, kmax + 1)
    l_values = []
    bics = []
    results = []

    for k in ks:
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None\
                or m is None\
                or S is None\
                or g is None\
                or log_likelihood is None:
            return None, None, None, None

        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        bic = p * np.log(n) - 2 * log_likelihood
        l_values.append(log_likelihood)
        bics.append(bic)
        results.append((pi, m, S))

    l_values = np.array(l_values)
    bics = np.array(bics)
    best_index = np.argmin(bics)
    best_k = ks[best_index]
    best_result = results[best_index]

    return best_k, best_result, l_values, bics
