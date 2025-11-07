#!/usr/bin/env python3
"""
    A function that performs the Monte Carlo algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
        Performs the Monte Carlo algorithm.

        Args:
            env (gym.Env): the environment instance.
            V (np.ndarray): a numpy.ndarray of shape (s,) containing the
                value estimate.
            policy: a function that takes in a state and returns the next
                action to take.
            episodes (int): the total number of episodes to train over.
            max_steps (int): the maximum number of steps per episode.
            alpha (float): the learning rate.
            gamma (float): the discount rate.

        Returns:
            V, the updated value estimate.
    """
    for _ in range(episodes):
        state, info = env.reset()
        episode = []
        step_count = 0
        done = False
        while not done and step_count < max_steps:
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, reward))
            state = next_state
            step_count += 1
            done = terminated or truncated

        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V
