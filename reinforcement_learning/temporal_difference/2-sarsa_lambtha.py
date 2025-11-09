#!/usr/bin/env python3
"""
    A function that performs SARSA(λ).
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        Performs the SARSA(λ) algorithm.

        Args:
            env (gym.Env): the environment instance.
            Q (np.ndarray): a numpy.ndarray of shape (s, a) containing the Q
                table.
            lambtha (float): the eligibility trace factor.
            episodes (int): the total number of episodes to train over.
            max_steps (int): the maximum number of steps per episode.
            alpha (float): the learning rate.
            gamma (float): the discount rate.
            epsilon (float): the initial threshold for epsilon greedy.
            min_epsilon (float): the minimum value that epsilon should decay to
            epsilon_decay (float): the decay rate for updating epsilon between
                episodes.

        Returns:
            Q, the updated Q table.
    """
    n_states, n_actions = Q.shape

    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        E = np.zeros((n_states, n_actions))

        for _ in range(max_steps):
            next_state, reward, done, *rest = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            td_target = reward +\
                gamma * Q[next_state, next_action] * (not done)
            delta = td_target - Q[state, action]
            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha
            state, action = next_state, next_action

            if done:
                break

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
