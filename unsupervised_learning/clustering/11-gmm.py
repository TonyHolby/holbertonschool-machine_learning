#!/usr/bin/env python3
"""
    A function that calculates a GMM from a dataset.
"""
import sklearn.mixture


def gmm(X, k):
    """
        Calculates a GMM from a dataset.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (n, d) containing the
            dataset.
            k (int): the number of clusters.

        Returns:
            pi, m, S, clss, bic:
                pi is a numpy.ndarray of shape (k,) containing the cluster
                priors.
                m is a numpy.ndarray of shape (k, d) containing the centroid
                means.
                S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices.
                clss is a numpy.ndarray of shape (n,) containing the cluster
                indices for each data point.
                bic is a numpy.ndarray of shape (kmax - kmin + 1) containing
                the BIC value for each cluster size tested.
    """
    if not isinstance(X, (list, tuple)) and not hasattr(X, 'shape'):
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
