#!/usr/bin/env python3
"""
    A script that defines a class named GaussianProcess that represents a
    noiseless 1D Gaussian process.
"""
import numpy as np


class GaussianProcess:
    """
        A class that represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            Initializes the Gaussian Process.

            Args:
                X_init (np.ndarray): a numpy.ndarray of shape (t, 1)
                    representing the inputs already sampled with the
                    black-box function.
                Y_init (np.ndarray): a numpy.ndarray of shape (t, 1)
                    representing the outputs of the black-box function for each
                    input in X_init.
                t (int): the number of initial samples.
                l (int): the length parameter for the kernel.
                sigma_f (int): the standard deviation given to the output of
                    the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
            Calculates the covariance kernel matrix between two matrices.

            Args:
                X1 (np.ndarray): a numpy.ndarray of shape (m, 1).
                X2 (np.ndarray): a numpy.ndarray of shape (n, 1).

            Returns:
                The covariance kernel matrix as a np.ndarray of shape (m, n).
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_euclidean_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)\
            - 2 * np.dot(X1, X2.T)
        covariance_kernel_matrix = (self.sigma_f ** 2)\
            * np.exp(-0.5 / self.l**2 * sq_euclidean_dist)

        return covariance_kernel_matrix

    def predict(self, X_s):
        """"
            Predicts the mean and standard deviation of points in a Gaussian
            process.

            Args:
                X_s (np.ndarray): a numpy.ndarray of shape (s, 1) containing
                all of the points whose mean and standard deviation should be
                calculated:
                    s is the number of sample points.

            Returns:
                mu, sigma:
                    mu is a numpy.ndarray of shape (s,) containing the mean for
                    each point in X_s, respectively.
                    sigma is a numpy.ndarray of shape (s,) containing the
                    variance for each point in X_s, respectively.
        """
        X_s = np.atleast_2d(X_s)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inverse = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inverse).dot(self.Y).reshape(-1)
        covariance_s = K_ss - K_s.T.dot(K_inverse).dot(K_s)
        sigma = np.diag(covariance_s)

        return mu, sigma
