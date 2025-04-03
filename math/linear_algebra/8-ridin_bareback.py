#!/usr/bin/env python3
""" A function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """
        Performs a multiplication of matrices.

        Parameters:
            mat1 (list of lists): The first 2D matrix.
            mat2 (list of lists): The second 2D matrix.

        Returns:
            A new matrix of the multiplication of mat1 and mat2
            or none otherwise.
    """
    if len(mat1[0]) != len(mat2):
        return None

    new_matrix = []

    for row in range(len(mat1)):
        sum_list = []
        for col in range(len(mat2[0])):
            sum_of_multiplications = 0
            for index in range(len(mat2)):
                sum_of_multiplications += mat1[row][index] * mat2[index][col]
            sum_list.append(sum_of_multiplications)
        new_matrix.append(sum_list)

    return new_matrix
