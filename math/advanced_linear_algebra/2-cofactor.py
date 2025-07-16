#!/usr/bin/env python3
""" A function that calculates the cofactor matrix of a matrix. """


def determinant(matrix):
    """
        Calculates the determinant of a matrix.

        Args:
            matrix (list): a list of lists whose determinant should be
            calculated.

        Returns:
            The determinant of matrix.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        det = 1

        return det

    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        det = matrix[0][0]

        return det

    if len(matrix) == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

        return det

    det = 0
    for col in range(len(matrix)):
        sign = (-1) ** col
        minor = []
        for row in matrix[1:]:
            new_row = row[:col] + row[col+1:]
            minor.append(new_row)
        det += sign * matrix[0][col] * determinant(minor)

    return det


def cofactor(matrix):
    """
        Calculates the cofactor matrix of a matrix.

        Args:
            matrix (list): a list of lists whose cofactor matrix should be
            calculated.

        Returns:
            The cofactor matrix of matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cofactor_matrix = []
    for index_row in range(len(matrix)):
        current_cofactor_row = []
        for index_col in range(len(matrix)):
            submatrix = []
            for index, row_list in enumerate(matrix):
                if index != index_row:
                    new_row = row_list[:index_col] +\
                        row_list[index_col + 1:]
                    submatrix.append(new_row)
            sign = (-1) ** (index_row + index_col)
            minor = determinant(submatrix)
            current_cofactor_row.append(sign * minor)
        cofactor_matrix.append(current_cofactor_row)

    return cofactor_matrix
