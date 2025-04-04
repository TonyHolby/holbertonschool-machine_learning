#!/usr/bin/env python3
import numpy as np
""" A function that calculates the shape of a numpy.ndarray """


def np_shape(matrix):
    """
        Calculates the shape of a numpy.ndarray.

        Parameters:
            matrix (numpy.ndarray): The input numpy array.

        Returns:
            A tuple of integers describing the shape of the matrix.
    """
    return np.shape(matrix)
