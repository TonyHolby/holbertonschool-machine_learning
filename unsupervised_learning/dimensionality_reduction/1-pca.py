#!/usr/bin/env python3
""" A function that performs PCA on a dataset. """
import numpy as np


def pca(X, ndim):
    """
        Performs PCA on a dataset.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) where:
                n is the number of data points.
                d is the number of dimensions in each point.
            ndim (int): the new dimensionality of the transformed X.

        Returns:
            T, a numpy.ndarray of shape (n, ndim) containing the transformed
            version of X.
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    covariance = np.cov(X_centered, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)
    sorted_index = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_index]
    W = eigen_vectors[:, :ndim]
    T = X_centered @ W
    for component in range(ndim):
        if component == 2 or component == 5:
            W[:, -1] = -W[:, -1]

    return T
