#!/usr/bin/env python3
"""
    A script that calculates the posterior probability that a patient who
    takes the drug will develop severe side effects, based on various
    hypothetical probabilities.
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

    return np.float64(likelihoods)


def intersection(x, n, P, Pr):
    """
        Calculates the intersection that a patient who takes the drug will
        develop severe side effects, based on various hypothetical
        probabilities.

        Args:
            x (int): the number of patients that develop severe side effects.
            n (int): the total number of patients observed.
            P (np.ndarray): a 1D numpy.ndarray containing the various
            hypothetical probabilities of developing severe side effects.
            Pr (np.ndarray): a 1D numpy.ndarray containing the prior beliefs
            of P.

        Returns:
            A 1D numpy.ndarray containing the intersection of obtaining
            x and n with each probability in P, respectively.
    """
    likelihood_value = likelihood(x, n, P)
    intersection = likelihood_value * Pr

    return np.float64(intersection)


def marginal(x, n, P, Pr):
    """
        Calculates the marginal probability that a patient who takes the drug
        will develop severe side effects, based on various hypothetical
        probabilities.

        Args:
            x (int): the number of patients that develop severe side effects.
            n (int): the total number of patients observed.
            P (np.ndarray): a 1D numpy.ndarray containing the various
            hypothetical probabilities of developing severe side effects.
            Pr (np.ndarray): a 1D numpy.ndarray containing the prior beliefs
            of P.

        Returns:
            The marginal probability of obtaining x and n.
    """
    intersection_value = intersection(x, n, P, Pr)
    marginal_probability = np.sum(intersection_value)

    return np.float64(marginal_probability)


def posterior(x, n, P, Pr):
    """
        Calculates the posterior probability for the various hypothetical
        probabilities of developing severe side effects given the data.

        Args:
            x (int): the number of patients that develop severe side effects.
            n (int): the total number of patients observed.
            P (np.ndarray): a 1D numpy.ndarray containing the various
            hypothetical probabilities of developing severe side effects.
            Pr (np.ndarray): a 1D numpy.ndarray containing the prior beliefs
            of P.

        Returns:
            The posterior probability of each probability in P given x and n,
            respectively.
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    intersection_value = intersection(x, n, P, Pr)
    marginal_probability = marginal(x, n, P, Pr)
    posterior_probability = intersection_value / marginal_probability

    return np.float64(posterior_probability)
