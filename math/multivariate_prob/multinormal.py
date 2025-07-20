#!/usr/bin/env python3
"""
    A script that represents a Multivariate Normal distribution
    and calculates the PDF at a data point.
"""
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

        d = data.shape[0]
        n = data.shape[1]

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """
            Calculates the PDF at a data point.

            Args:
                x is a numpy.ndarray of shape (d, 1) containing the data point
                whose PDF should be calculated:
                    d is the number of dimensions of the Multinomial instance.

            Returns:
                The value of the PDF.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        determinant_covariance = np.linalg.det(self.cov)
        inverse_covariance = np.linalg.inv(self.cov)
        x_centered = x - self.mean

        mahalanobis_distance = -0.5 * (
            x_centered.T @ inverse_covariance @ x_centered)
        normalization_constant = 1 / np.sqrt(
            ((2 * np.pi) ** self.d) * determinant_covariance)

        pdf_value = float(
            normalization_constant * np.exp(mahalanobis_distance))

        return pdf_value
