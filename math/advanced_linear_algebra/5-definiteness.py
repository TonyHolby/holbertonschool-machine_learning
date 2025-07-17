#!/usr/bin/env python3
""" A function that calculates the definiteness of a matrix. """
import numpy as np


def definiteness(matrix):
    """
        Calculates the definiteness of a matrix.

        Args:
            matrix (numpy.ndarray): a numpy.ndarray of shape (n, n) whose
            definiteness should be calculated.

        Returns:
            The categorie of definiteness of the matrix: Positive definite,
            Positive semi-definite, Negative semi-definite, Negative definite,
            or Indefinite.
            Returns None, if matrix does not fit any of these categories.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
