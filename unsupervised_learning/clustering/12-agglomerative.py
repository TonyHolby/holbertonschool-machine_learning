#!/usr/bin/env python3
"""
    A function that performs agglomerative clustering on a dataset with Ward
    linkage and shows the dendrogram.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
        Performs agglomerative clustering on a dataset with Ward linkage
        and shows the dendrogram.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the
            dataset.
            dist (int or float): the maximum cophenetic distance for all
            clusters.

        Returns:
            clss, a numpy.ndarray of shape (n,) containing the cluster indices
            for each data point.
    """
    if not isinstance(X, (list, tuple)) and not hasattr(X, 'shape'):
        return None

    if not isinstance(dist, (int, float)) or dist <= 0:
        return None

    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
