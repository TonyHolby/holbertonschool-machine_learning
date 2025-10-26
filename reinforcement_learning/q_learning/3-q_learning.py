#!/usr/bin/env python3
"""
    A function that performs Q-learning.
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        Performs Q-learning.

        Args:
            env: the FrozenLakeEnv instance.
            Q (np.ndarray): a numpy.ndarray containing the Q-table.
            episodes (int): the total number of episodes to train over.
            max_steps (int): the maximum number of steps per episode.
            alpha (float): the learning rate.
            gamma (float): the discount rate.
            epsilon (int): the initial threshold for epsilon greedy.
            min_epsilon (float): the minimum value that epsilon should decay
                to.
            epsilon_decay (float): the decay rate for updating epsilon between
                episodes.

        Returns:
            Q, total_rewards:
                Q is the updated Q-table.
                total_rewards is a list containing the rewards per episode.
    """
    total_rewards = []
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
