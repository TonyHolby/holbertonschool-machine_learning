#!/usr/bin/env python3
"""
    A function that performs the forward algorithm for a hidden markov model.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
        Performs the forward algorithm for a hidden markov model.

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
            P, F, or None, None on failure:
                P is the likelihood of the observations given the model.
                F is a numpy.ndarray of shape (N, T) containing the forward
                path probabilities:
                    F[i, j] is the probability of being in hidden state i at
                    time j given the previous observations.
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None

        F = np.zeros((N, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
        for t in range(1, T):
            for j in range(N):
                F[j, t] = np.sum(F[:, t-1] * Transition[:, j]
                                 ) * Emission[j, Observation[t]]

        P = np.sum(F[:, -1])

        return P, F

    except Exception:
        return None, None
