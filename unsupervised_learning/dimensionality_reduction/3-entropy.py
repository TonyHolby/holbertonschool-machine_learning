#!/usr/bin/env python3
"""
    A function that calculates the Shannon entropy and P affinities relative
    to a data point.
"""
import numpy as np


def HP(Di, beta):
    """
        Calculates the Shannon entropy and P affinities relative to a data
        point.

        Args:
            Di is a numpy.ndarray of shape (n - 1,) containing the pariwise
            distances between a data point and all other points except
            itself:
                n is the number of data points
            beta is a numpy.ndarray of shape (1,) containing the beta value
            for the Gaussian distribution.

        Returns:
            (Hi, Pi):
                Hi: the Shannon entropy of the points
                Pi: a numpy.ndarray of shape (n - 1,) containing the P
                affinities of the points.
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    if sum_Pi == 0:
        Pi = np.full_like(Di, 1 / Di.shape[0])
    else:
        Pi /= sum_Pi

    Hi = -np.sum(Pi * np.log2(Pi + 1e-7))

    return Hi, Pi
