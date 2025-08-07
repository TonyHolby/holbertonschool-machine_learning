#!/usr/bin/env python3
"""
    A function that calculates the most likely sequence of hidden states for
    a hidden markov model.
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
        Calculates the most likely sequence of hidden states for a hidden
        markov model.

        Args:
            Observation (np.ndarray): a numpy.ndarray of shape (T,) that
                contains the index of the observation (T: number of
                observations).
            Emission (np.ndarray): a numpy.ndarray of shape (N, M) containing
                the emission probability of a specific observation given a
                hidden state (Emission[i, j]: probability of observing j given
                the hidden state i, N: number of hidden states, M is the number
                of all possible observations).
            Transition (np.ndarray): a 2D numpy.ndarray of shape (N, N)
                containing the transition probabilities (Transition[i, j]:
                probability of transitioning from the hidden state i to j).
            Initial (np.ndarray): a numpy.ndarray of shape (N, 1) containing
                the probability of starting in a particular hidden state.

        Returns:
            path, P, or None, None on failure:
                path is the a list of length T containing the most likely
                sequence of hidden states.
                P is the probability of obtaining the path sequence.
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        V = np.zeros((N, T))
        B = np.zeros((N, T), dtype=int)

        V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
        B[:, 0] = 0

        for t in range(1, T):
            for j in range(N):
                probabilities = V[:, t - 1]\
                    * Transition[:, j]\
                    * Emission[j, Observation[t]]
                V[j, t] = np.max(probabilities)
                B[j, t] = np.argmax(V[:, t - 1] * Transition[:, j])

        P = np.max(V[:, -1])
        last_state = np.argmax(V[:, -1])

        path = [0] * T
        path[-1] = last_state
        for t in range(T - 2, -1, -1):
            path[t] = B[path[t + 1], t + 1]

        return path, P

    except Exception:
        return None, None
