#!/usr/bin/env python3
""" A function that performs K-means on a dataset. """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
        Performs K-means on a dataset.

        Args:
            X is (np.ndarray): numpy.ndarray of shape (n, d) containing the
            dataset that will be used for K-means clustering:
                n is the number of data points.
                d is the number of dimensions for each data point.
            k (int): a positive integer containing the number of clusters.
            iterations (int): a positive integer containing the maximum
            number of iterations that should be performed.

        Returns:
            C, clss, or None, None on failure:
                C is a numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster.
                clss is a numpy.ndarray of shape (n,) containing the index
                of the cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    d = X.shape[1]

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    C = np.random.uniform(min_values, max_values, (k, d))

    for i in range(iterations):
        euclidean_distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(euclidean_distances, axis=1)
        new_centroids = C.copy()
        for current_cluster in range(k):
            points = X[clss == current_cluster]
            if points.shape[0] == 0:
                new_centroids[current_cluster] = np.random.uniform(
                    low=np.min(X, axis=0),
                    high=np.max(X, axis=0))
            else:
                new_centroids[current_cluster] = np.mean(points, axis=0)

        if np.allclose(C, new_centroids):
            break

    return C, clss
