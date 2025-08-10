#!/usr/bin/env python3
"""
    A function that  performs the Baum-Welch algorithm for a hidden markov
    model.
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
        Performs the Baum-Welch algorithm for a hidden markov model.

        Args:
            Observations (np.ndarray): a numpy.ndarray of shape (T,) that
                contains the index of the observation (T is the number of
                observations).
            Transition (np.ndarray): a numpy.ndarray of shape (M, M) that
                contains the initialized transition probabilities (M is the
                number of hidden states).
            Emission (np.ndarray): a numpy.ndarray of shape (M, N) that
                contains the initialized emission probabilities (N: the number
                of output states).
            Initial (np.ndarray): a numpy.ndarray of shape (M, 1) that contains
                the initialized starting probabilities.
            iterations (int): the number of times expectation-maximization
                should be performed.

        Returns:
            The converged Transition, Emission, or None, None on failure.
    """
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        for _ in range(iterations):
            alpha = np.zeros((M, T))
            alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
            for t in range(1, T):
                for j in range(M):
                    alpha[j, t] = np.sum(alpha[:, t - 1] * Transition[:, j])\
                        * Emission[j, Observations[t]]

            beta = np.zeros((M, T))
            beta[:, -1] = 1
            for t in range(T - 2, -1, -1):
                for i in range(M):
                    beta[i, t] = np.sum(Transition[i, :]
                                        * Emission[:, Observations[t + 1]]
                                        * beta[:, t + 1])

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denominator = np.sum(alpha[:, t]
                                     * (Transition
                                        @ (Emission[:, Observations[t+1]]
                                           * beta[:, t+1])))

                for i in range(M):
                    numerator = alpha[i, t]\
                        * Transition[i, :]\
                        * Emission[:, Observations[t + 1]] * beta[:, t + 1]
                    if denominator != 0:
                        xi[i, :, t] = numerator / denominator
                    else:
                        xi[i, :, t] = 0

            gamma = np.sum(xi, axis=1)
            last_gamma = (alpha[:, -1] * beta[:, -1])\
                / np.sum(alpha[:, -1] * beta[:, -1])
            gamma = np.hstack((gamma, last_gamma.reshape(-1, 1)))

            Initial = gamma[:, 0].reshape(-1, 1)
            Transition = np.sum(xi, axis=2)\
                / np.sum(gamma[:, :-1], axis=1, keepdims=True)

            for k in range(N):
                mask = (Observations == k)
                Emission[:, k] = np.sum(gamma[:, mask], axis=1)
            Emission /= np.sum(gamma, axis=1, keepdims=True)

        return Transition, Emission

    except Exception:
        return None, None
