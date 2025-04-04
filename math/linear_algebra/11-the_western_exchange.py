#!/usr/bin/env python3
import numpy as np
""" A function that transposes a matrix """


def np_transpose(matrix):
    """
        Transposes a matrix.

        Parameters:
            matrix (numpy.ndarray): The input numpy array.

        Returns:
            A new numpy.ndarray.
    """
    transposed_matrix = matrix.T

    return transposed_matrix
