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
    states_space = env.observation_space.n
    actions_space = env.action_space.n
    Q = np.zeros((states_space, actions_space))

    return Q
