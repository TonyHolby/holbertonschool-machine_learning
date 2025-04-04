#!/usr/bin/env python3
""" A function that performs matrix multiplication """

import numpy as np


def np_matmul(mat1, mat2):
    """
        Performs multiplication of the matrices mat1 and mat2.

        Parameters:
            mat1 (numpy.ndarray): The first numpy array.
            mat2 (numpy.ndarray): The second numpy array.

        Returns:
            A new numpy array of the multiplication of mat1 and mat2.
    """
    new_numpy_array = np.matmul(mat1, mat2)

    return new_numpy_array
