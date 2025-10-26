#!/usr/bin/env python3
"""
    A function that has the trained agent play an episode.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
        Has the trained agent play an episode.

        Args:
            env: the FrozenLakeEnv instance.
            Q (np.ndarray): a numpy.ndarray containing the Q-table.
            max_steps (int): the maximum number of steps per episode.

        Returns:
            The total rewards for the episode and a list of rendered outputs
            representing the board state at each step.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []
    done = False
    initial_render = env.render()
    rendered_outputs.append(initial_render)

    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_rewards += reward
        current_render = env.render()
        rendered_outputs.append(current_render)
        state = next_state

        if done:
            break

    return total_rewards, rendered_outputs
