#!/usr/bin/env python3
"""
    A function that calculates the likelihood that a patient who takes the
    drug will develop severe side effects, based on various hypothetical
    probabilities.
"""
import numpy as np


def likelihood(x, n, P):
    """
        Calculates the likelihood that a patient who takes the drug will
        develop severe side effects, based on various hypothetical
        probabilities.

        Args:
            x (int): the number of patients that develop severe side effects.
            n (int): the total number of patients observed.
            P (np.ndarray): a 1D numpy.ndarray containing the various
            hypothetical probabilities of developing severe side effects.

        Returns:
            A 1D numpy.ndarray containing the likelihood of obtaining the
            data, x and n, for each probability in P, respectively.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    n_factorial = 1
    for i in range(1, n + 1):
        n_factorial *= i

    x_factorial = 1
    for i in range(1, x + 1):
        x_factorial *= i

    n_minus_x_factorial = 1
    for i in range(1, (n - x) + 1):
        n_minus_x_factorial *= i

    combination_n_x = n_factorial // (x_factorial * n_minus_x_factorial)

    likelihoods = np.array(combination_n_x * (P ** x) * ((1 - P) ** (n - x)))

    return likelihoods
