#!/usr/bin/env python3
"""
    A function that performs K-means on a dataset.
"""
import sklearn.cluster


def kmeans(X, k):
    """
        Performs K-means on a dataset.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the
            dataset.
            k (int): the number of clusters.

        Returns:
            C, clss:
                C is a numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster.
                clss is a numpy.ndarray of shape (n,) containing the index of
                the cluster in C that each data point belongs to.
    """
    if not isinstance(X, (list, tuple)) and not hasattr(X, 'shape'):
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    kmeans_model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(X)

    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
