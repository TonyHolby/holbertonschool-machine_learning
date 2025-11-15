#!/usr/bin/env python3
"""Training module for policy gradient with optional visualization"""
import numpy as np
import matplotlib.pyplot as plt
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training using policy gradient.

    Args:
        env: environment object.
        nb_episodes: number of training episodes.
        alpha: learning rate.
        gamma: discount factor.
        show_result: if True, render an episode every 1000 episodes (non-intrusive)

    Returns:
        List of episode scores.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Weight initialization
    weight = np.random.rand(state_size, action_size)

    scores = []

    # Reset BEFORE training loop (required for reproducibility)
    state, _ = env.reset()

    # Matplotlib window for visualization (if enabled)
    fig, ax = None, None
    if show_result:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))

    for episode in range(nb_episodes):
        episode_rewards = []
        episode_gradients = []
        done = False
        truncated = False

        # Training loop for the episode
        while not done and not truncated:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            episode_rewards.append(reward)
            episode_gradients.append(gradient)
            state = next_state

        # Score
        score = sum(episode_rewards)
        scores.append(score)

        # Compute discounted rewards
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

        # Weight update
        for t in range(len(episode_gradients)):
            weight += alpha * episode_gradients[t] * discounted_rewards[t]

        print(f"Episode: {episode} Score: {score}")

        # Reset environment at end of episode (important)
        state, _ = env.reset()

        # --------------------------------------------------
        # OPTIONAL VISUALIZATION (non-intrusive)
        # --------------------------------------------------
        if show_result and episode % 1000 == 0:
            try:
                frame = env.render()
                if frame is not None:
                    ax.clear()
                    ax.imshow(frame)
                    ax.axis('off')
                    ax.set_title(f"Episode {episode} - Score {score}")
                    plt.pause(0.001)
            except:
                pass

    # Close window
    if show_result:
        plt.ioff()
        plt.close(fig)

    return scores
