#!/usr/bin/env python3
"""
    A function that determines the steady state probabilities of a regular
    markov chain.
"""
import numpy as np


def regular(P):
    """
        Determines the steady state probabilities of a regular markov chain.

        Args:
            P (np.ndarray): a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix:
                P[i, j] is the probability of transitioning from state i to
                state j.
                n is the number of states in the markov chain.

        Returns:
            A numpy.ndarray of shape (1, n) containing the steady state
            probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray):
        return None

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    try:
        k = 100
        matrix_power = np.linalg.matrix_power(P, k)
        if not np.all(matrix_power > 0):
            return None
    except Exception:
        return None

    identiy_matrix = np.eye(n)
    matrix_A = P.T - identiy_matrix
    matrix_A = np.vstack([matrix_A, np.ones((1, n))])
    b = np.zeros((n + 1,))
    b[-1] = 1

    try:
        steady_state = np.linalg.lstsq(matrix_A, b, rcond=None)[0]
        return steady_state.reshape(1, n)
    except Exception:
        return None
