#!/usr/bin/env python3
"""
    A function that computes the policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weight):
    """
        Computes the policy with a weight of a matrix.

        Args:
            matrix (np.ndarray): a numpy ndarray representing the state.
            weight (np.ndarray): a numpy ndarray of weights.

        Returns:
            An array of action probabilities for a given state.
    """
    scores = np.dot(matrix, weight)
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores) / np.sum(
        np.exp(scores), axis=1, keepdims=True)

    return probabilities


def policy_gradient(state, weight):
    """
        Computes the Monte-Carlo policy gradient based on a state and weight
        matrix.

        Args:
            state (np.ndarray): matrix representing the current observation of
                the environment.
            weight (np.ndarray): matrix of random weight.

        Returns:
            The action and the gradient (in this order).
    """
    state = state.reshape(1, -1)
    action_probabilities = policy(state, weight)
    action = np.random.choice(
        len(action_probabilities[0]), p=action_probabilities[0])
    indicator = np.zeros_like(action_probabilities)
    indicator[0, action] = 1
    gradient = state.T.dot(indicator - action_probabilities)

    return action, gradient
