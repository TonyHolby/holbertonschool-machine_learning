#!/usr/bin/env python3
"""
    A function that initializes the Q-table.
"""
import numpy as np


def q_init(env):
    """
        Initializes the Q-table.

        Args:
            env: the FrozenLakeEnv instance.

        Returns:
            The Q-table as a numpy.ndarray of zeros.
    """
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    Q = np.zeros((number_of_states, number_of_actions))

    return Q
