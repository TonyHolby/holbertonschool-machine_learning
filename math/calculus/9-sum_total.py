#!/usr/bin/env python3
""" A function that calculates Sigma i squared for i from 1 to n """


def summation_i_squared(n):
    """
        Calculates Sigma i squared for i from 1 to n.

        Parameter:
            n (int): The stopping condition.

        Return:
            None, if n is not an integer or < 0.
            The integer value of the sum, else.
    """
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    sigma = n**2 + summation_i_squared(n - 1)

    return sigma
