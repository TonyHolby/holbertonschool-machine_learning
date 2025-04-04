#!/usr/bin/env python3
import numpy as np
"""
    A function that performs element-wise addition, subtraction,
    multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """
        Performs element-wise addition, subtraction,
        multiplication, and division.

        Parameters:
            mat1 (numpy.ndarray): The first numpy array.
            mat2 (numpy.ndarray): The second numpy array.

        Returns:
            A tuple containing the element-wise sum, difference,
            product, and quotient, respectively.
    """
    add, sub, mul, div = mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2

    return add, sub, mul, div
