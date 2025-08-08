#!/usr/bin/env python3
"""
    A function that performs the backward algorithm for a hidden markov model.
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
        Performs the backward algorithm for a hidden markov model.

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
            P, B, or None, None on failure:
                Pis the likelihood of the observations given the model.
                B is a numpy.ndarray of shape (N, T) containing the backward
                path probabilities:
                    B[i, j] is the probability of generating the future
                    observations from hidden state i at time j.
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        B = np.zeros((N, T))
        B[:, -1] = 1
        for t in range(T - 2, -1, -1):
            for i in range(N):
                B[i, t] = np.sum(Transition[i, :] *
                                 Emission[:, Observation[t + 1]] * B[:, t + 1])

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
