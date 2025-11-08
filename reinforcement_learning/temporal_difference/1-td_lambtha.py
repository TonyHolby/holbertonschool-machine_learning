#!/usr/bin/env python3
"""
    A function that performs the TD(λ) algorithm.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
        Performs the TD(λ) algorithm.

        Args:
            env (gym.Env): the environment instance.
            V (np.ndarray): a numpy.ndarray of shape (s,) containing the
                value estimate.
            policy: a function that takes in a state and returns the next
                action to take.
            lambtha (float): the eligibility trace factor.
            episodes (int): the total number of episodes to train over.
            max_steps (int): the maximum number of steps per episode.
            alpha (float): the learning rate.
            gamma (float): the discount rate.

        Returns:
            V, the updated value estimate.
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        E = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, *rest = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            td_target = reward + gamma * V[next_state] * (not done)
            delta = td_target - V[state]
            E[state] += 1
            V += alpha * delta * E
            E *= gamma * lambtha
            state = next_state

            if done:
                break

    return V
