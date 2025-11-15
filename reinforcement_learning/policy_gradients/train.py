#!/usr/bin/env python3
"""
    A function that implements a full training.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.0045, gamma=0.999):
    """
        Implements a full training.

        Args:
            env: initial environment
            nb_episodes: number of episodes used for training
            alpha: the learning rate
            gamma: the discount factor

        Returns:
            All values of the score (sum of all rewards during one episode
            loop)
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    weight = np.random.rand(state_size, action_size)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_gradients = []
        episode_actions = []
        done = False
        truncated = False

        while not done and not truncated:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            episode_rewards.append(reward)
            episode_gradients.append(gradient)
            episode_actions.append(action)

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        discounted_rewards = []
        cumulative = 0
        for reward in reversed(episode_rewards):
            cumulative = reward + gamma * cumulative
            discounted_rewards.insert(0, cumulative)

        discounted_rewards = np.array(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (
                discounted_rewards - np.mean(discounted_rewards)
                ) / (np.std(discounted_rewards) + 1e-8)

        for t in range(len(episode_gradients)):
            weight += alpha * episode_gradients[t] * discounted_rewards[t]

        print(f"Episode: {episode} Score: {score}")

    return scores


if __name__ == "__main__":
    train()
