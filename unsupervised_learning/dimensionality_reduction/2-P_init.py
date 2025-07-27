#!/usr/bin/env python3
"""
    A function that initializes all variables required to calculate the
    P affinities in t-SNE.
"""
import numpy as np


def P_init(X, perplexity):
    """
        Initializes all variables required to calculate the P affinities
        in t-SNE.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the
            dataset to be transformed by t-SNE:
                n is the number of data points.
                d is the number of dimensions in each point.
            perplexity (float): the perplexity that all Gaussian distributions
            should have.

        Returns:
            (D, P, betas, H):
                D: a numpy.ndarray of shape (n, n) that calculates the squared
                pairwise distance between two data points.
                P: a numpy.ndarray of shape (n, n) initialized to all 0's that
                will contain the P affinities.
                betas: a numpy.ndarray of shape (n, 1) initialized to all 1's
                that will contain all of the beta values:
                    beta_{i} = 1 / (2 * (sigma_{i} ** 2))
                H is the Shannon entropy for perplexity perplexity with a base
                of 2.
    """
    n = X.shape[0]
    squared_norms = np.sum(X ** 2, axis=1)
    D = squared_norms[:, np.newaxis] + squared_norms[np.newaxis, :] \
        - 2 * np.dot(X, X.T)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
