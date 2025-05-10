#!/usr/bin/env python3
"""
    A function that normalizes an unactivated output of a neural network using
    batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Normalizes an unactivated output of a neural network using batch
        normalizationy.

        Args:
            Z (np.ndarray): a numpy.ndarray of shape (m, n) that should be
            normalized :
                m is the number of data points.
                n is the number of features in Z.
            gamma (np.ndarray): a numpy.ndarray of shape (1, n) containing
                the scales used for batch normalization.
            beta (np.ndarray): a numpy.ndarray of shape (1, n) containing
                the offsets used for batch normalization.
            epsilon (float): a small number used to avoid division by zero.

        Returns:
            The normalized Z matrix.
    """
    m = Z.shape[0]
    mean = (1 / m) * np.sum(Z, axis=0, keepdims=True)
    variance = (1 / m) * np.sum((Z - mean) ** 2, axis=0, keepdims=True)
    normalized_Z = (Z - mean) / np.sqrt(variance + epsilon)
    scaled_Z = gamma * normalized_Z + beta

    return scaled_Z
