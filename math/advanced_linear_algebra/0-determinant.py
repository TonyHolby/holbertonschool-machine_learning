#!/usr/bin/env python3
""" A function that calculates the determinant of a matrix. """


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

    if matrix == []:
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
