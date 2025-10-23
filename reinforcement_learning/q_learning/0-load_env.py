#!/usr/bin/env python3
"""
    A function that loads the pre-made FrozenLakeEnv evnironment from
    gymnasium.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        Loads the pre-made FrozenLakeEnv evnironment from gymnasium.

        Args:
            desc (list of lists or None): a list of lists containing a custom
                description of the map to load for the environment.
            map_name (str or None): a string containing the pre-made map to
                load.
            is_slippery (bool): a boolean to determine if the ice is slippery.

        Returns:
            The FrozenLake environment.
    """
    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
