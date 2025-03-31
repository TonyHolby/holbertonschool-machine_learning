#!/usr/bin/env python3
import numpy as np
""" A function to calculate the shape of a matrix """


def matrix_shape(matrix):
    """
        Returns the shape of a matrix as a list of integers.

        matrix: The input matrix (list of lists).
    """
    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []

    return shape
