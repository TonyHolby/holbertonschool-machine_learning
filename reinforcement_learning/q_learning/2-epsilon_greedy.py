#!/usr/bin/env python3
"""
    A function that uses epsilon-greedy to determine the next action.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
        Uses epsilon-greedy to determine the next action.

        Args:
            Q (np.ndarray): a numpy.ndarray containing the Q-table.
            state (int): the current state.
            epsilon (float): the epsilon to use for the calculation.

        Returns:
            The next action index.
    """
    random_number = np.random.uniform(0, 1)
    action_space = Q.shape[1]
    if random_number < epsilon:
        action = np.random.randint(0, action_space)
    else:
        action = np.argmax(Q[state])

    return action
