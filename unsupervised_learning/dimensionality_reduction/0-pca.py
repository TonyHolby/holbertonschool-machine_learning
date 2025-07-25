#!/usr/bin/env python3
""" A function that performs PCA on a dataset. """
import numpy as np


def pca(X, var=0.95):
    """
        Performs PCA on a dataset.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) where:
                n is the number of data points.
                d is the number of dimensions in each point.
            var (float) the fraction of the variance that the PCA
            transformation should maintain.

        Returns:
            The weights matrix, W, that maintains var fraction of
            X's original variance.
    """
    covariance_matrix = np.cov(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    total_variance = np.sum(eigen_values)
    cumulative_variance = np.cumsum(eigen_values) / total_variance
    number_of_components = np.searchsorted(cumulative_variance, var) + 2
    W = eigen_vectors[:, :number_of_components]

    return W
