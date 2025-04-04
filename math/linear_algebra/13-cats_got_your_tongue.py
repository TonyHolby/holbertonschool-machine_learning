#!/usr/bin/env python3
import numpy as np
""" A function that concatenates two matrices along a specific axis """


def np_cat(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis.

        Parameters:
            mat1 (numpy.ndarray): The first numpy array.
            mat2 (numpy.ndarray): The second numpy array.
            axis (0 or 1): The specific axis. 0 as default.

        Returns:
            A new numpy.ndarray.
    """
    new_np_array = np.hstack((mat1, mat2))

    return new_np_array
