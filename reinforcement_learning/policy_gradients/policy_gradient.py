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
