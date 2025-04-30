#!/usr/bin/env python3
"""
    A function named one_hot_decode that converts a one-hot matrix
    into a vector of labels.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a one-hot matrix into a vector of labels.

        Args:
            one_hot (np.ndarray): A one-hot encoded numpy.ndarray
                with shape (classes, m).

        Returns:
            A numpy.ndarray with shape (m, ) containing the numeric
            labels for each example, or None on failure.
    """
    try:
        oh_decoding = np.argmax(one_hot, axis=0)

        return oh_decoding

    except Exception:
        return None
