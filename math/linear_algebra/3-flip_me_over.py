#!/usr/bin/env python3
""" A function that returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """
        Returns the transpose of a 2D matrix.

        matrix: The input matrix (list of lists).
    """
    new_matrix = [list(row) for row in zip(*matrix)]

    return new_matrix
