#!/usr/bin/env python3
""" A function that concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis.

        Parameters:
            mat1 (list of lists): The first 2D matrix.
            mat2 (list of lists): The second 2D matrix.
            axis (0 or 1): The specific axis in a 2D matrix. 0 as default.

        Returns:
            A new matrix of the concatenation of mat1 and mat2.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        new_matrix = [row[:] for row in mat1]
        new_matrix += [row[:] for row in mat2]

        return new_matrix

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        new_matrix = [row1 + row2
                      for row1, row2
                      in zip(mat1, mat2)]

        return new_matrix

    return None
