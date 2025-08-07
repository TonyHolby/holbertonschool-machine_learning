#!/usr/bin/env python3
"""
    A function that determines if a markov chain is absorbing.
"""
import numpy as np


def absorbing(P):
    """
        Determines if a markov chain is absorbing.

        Args:
            P is a is a square 2D numpy.ndarray of shape (n, n) representing
            the standard transition matrix:
                P[i, j] is the probability of transitioning from state i to
                state j.
                n is the number of states in the markov chain.

        Returns:
            True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray):
        return False

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    absorbing_states = [i for i in range(n) if P[i, i] == 1
                        and np.all(P[i, :] == np.eye(n)[i])]

    if not absorbing_states:
        return False

    state_access_probs = np.copy(P)
    for _ in range(n):
        state_access_probs = np.matmul(state_access_probs, P)

    for i in range(n):
        if i not in absorbing_states:
            if not any(state_access_probs[i, j] > 0 for j in absorbing_states):
                return False

    return True
