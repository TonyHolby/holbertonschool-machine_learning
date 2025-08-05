#!/usr/bin/env python3
"""
    A function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations.
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
        Determines the probability of a markov chain being in a particular
        state after a specified number of iterations.

        Args:
            P (np.ndarray): a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix:
                P[i, j] is the probability of transitioning from state i to
                state j.
                n is the number of states in the markov chain.
            s (np.ndarray): a numpy.ndarray of shape (1, n) representing the
            probability of starting in each state.
            t (int): the number of iterations that the markov chain has been
            through.

        Returns:
            A numpy.ndarray of shape (1, n) representing the probability of
            being in a specific state after t iterations, or None on failure.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    if s.shape != (1, P.shape[0]):
        return None

    if not isinstance(t, int) or t < 1:
        return None

    new_state = s.copy()
    for _ in range(t):
        new_state = np.dot(new_state, P)

    return new_state
