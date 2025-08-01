#!/usr/bin/env python3
"""
    A function that performs the expectation maximization for a GMM.
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
        Performs the expectation maximization for a GMM.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            set.
            k (int): a positive integer containing the number of clusters.
            iterations (int): a positive integer containing the maximum number
            of iterations for the algorithm.
            tol (float): a non-negative float containing tolerance of the log
            likelihood, used to determine early stopping i.e. if the difference
            is less than or equal to tol you should stop the algorithm.
            verbose (bool): a boolean that determines if you should print
            information about the algorithm.

        Returns:
            pi, m, S, g, l, or None, None, None, None, None on failure:
                pi is a numpy.ndarray of shape (k,) containing the priors for
                each cluster.
                m is a numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster.
                S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for each cluster.
                g is a numpy.ndarray of shape (k, n) containing the
                probabilities for each data point in each cluster.
                l is the log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    previous_l = float('-inf')

    for i in range(iterations):
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

        delta = abs(log_likelihood - previous_l)
        if delta <= tol:
            if verbose and i % 10 != 0:
                print(f"Log Likelihood after {i} "
                      f"iterations: {log_likelihood:.5f}")
            break

        previous_l = log_likelihood
        pi, m, S = maximization(X, g)

    else:
        if verbose and (iterations - 1) % 10 != 0:
            print(f"Log Likelihood after {iterations} "
                  f"iterations: {log_likelihood:.5f}")

    g, log_likelihood = expectation(X, pi, m, S)

    return pi, m, S, g, log_likelihood
