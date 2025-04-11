#!/usr/bin/env python3
""" A function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """
        Calculates the integral of a polynomial.

        Parameters:
            poly (list): The list of coefficients representing a polynomial.
            C (int): An integer representing the integration constant.

        Return:
            None, if poly or C are not valid.
            A new list of coefficients representing
            the integral of the polynomial, else.
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if not poly:
        return None

    if poly == [0]:
        return [C]

    n = 1
    integral_coef = 0
    new_list = [C]
    number = 1

    for number in range(len(poly)):
        integral_coef = poly[number] * (1 / n)
        if integral_coef == int(integral_coef):
            new_list.append(int(integral_coef))
        else:
            new_list.append(integral_coef)
        n += 1

    return new_list
