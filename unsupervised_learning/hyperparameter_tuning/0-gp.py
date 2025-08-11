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
        squared_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)\
            - 2 * np.dot(X1, X2.T)
        cov_kernel_matrix = (self.sigma_f ** 2)\
            * np.exp(-0.5 / self.l**2 * squared_dist)

        return cov_kernel_matrix
