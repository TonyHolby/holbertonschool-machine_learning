#!/usr/bin/env python3
""" A class MultiNormal that represents a Multivariate Normal distribution. """
import numpy as np


class MultiNormal:
    """
        A class MultiNormal that represents a Multivariate Normal distribution.
    """
    def __init__(self, data):
        """
            Initializes a Multivariate Normal distribution.

            Args:
                data (numpy.ndarray): a numpy.ndarray of shape (d, n)
                containing the data set:
                    n is the number of data points.
                    d is the number of dimensions in each data point.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n = data.shape[1]

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)
