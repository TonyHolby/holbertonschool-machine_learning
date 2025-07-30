#!/usr/bin/env python3
"""
    A function that calculates the probability density function of a Gaussian
    distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
        Calculates the probability density function of a Gaussian distribution.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the data
            points whose PDF should be evaluated.
            m (np.ndarray): a numpy.ndarray of shape (d,) containing the mean
            of the distribution.
            S (np.ndarray): a numpy.ndarray of shape (d, d) containing the
            covariance of the distribution.

        Returns:
            P, or None on failure:
                P is a numpy.ndarray of shape (n,) containing the PDF values
                for each data point.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or m.shape != (X.shape[1],):
        return None

    if not isinstance(S, np.ndarray) or S.shape != (X.shape[1], X.shape[1]):
        return None

    d = X.shape[1]

    try:
        determinant_S = np.linalg.det(S)
        inverse_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    exponential_term = -0.5 * np.sum((X - m) @ inverse_S * (X - m), axis=1)
    normalization_factor = 1.0 / np.sqrt((2 * np.pi) ** d * determinant_S)
    P = normalization_factor * np.exp(exponential_term)
    P = np.maximum(P, 1e-300)

    return P
