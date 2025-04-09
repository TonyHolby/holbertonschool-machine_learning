#!/usr/bin/env python3
""" A function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """
        Calculates the derivative of a polynomial.

        Parameter:
            poly (list): The list of coefficients representing a polynomial.

        Return:
            None, if poly is not valid.
            [0], if the derivative is 0.
            A new list of coefficients representing
            the derivative of the polynomial, else.
    """
    if not isinstance(poly, list):
        return None

    if poly == []:
        return None

    n = 0
    derivative_coef = 0
    new_list = []

    for number in poly:
        if not isinstance(number, int):
            return None

        if poly == [0]:
            return [0]

        derivative_coef = number * n

        if n > 0:
            new_list.append(derivative_coef)
        n += 1

    return new_list
