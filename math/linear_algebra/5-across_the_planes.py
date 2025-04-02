#!/usr/bin/env python3
""" A function that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """
        Adds two matrices elements-wise.

        Parameters:
            mat1 (list of lists): The first 2D matrix.
            mat2 (list of lists): The second 2D matrix.

        Returns:
            A new matrix of the sum of mat1 and mat2.
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for row1, row2 in zip(mat1, mat2):
            if len(row1) != len(row2):
                return None
            new_matrix.append([number1 + number2
                               for number1, number2
                               in zip(row1, row2)])
        return new_matrix

    return None
