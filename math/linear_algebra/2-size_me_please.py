#!/usr/bin/env python3
import numpy as np
""" A function to calculate the shape of a matrix """


def matrix_shape(matrix):
    """
        matrix : the matrix of integers.
    """
    shape_of_matrix = np.array(matrix)
    return shape_of_matrix.shape
