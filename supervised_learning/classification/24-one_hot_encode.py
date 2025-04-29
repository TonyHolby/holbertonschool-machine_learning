#!/usr/bin/env python3
"""
    A function named one_hot_encode that converts a numeric
    label vector into a one-hot matrix.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix.

        Args:
            Y (np.ndarray): A numpy ndarray of shape (m,) containing
                the correct labels for the input data.
            classes (int): The maximum number of classes found in Y.

        Returns:
            A one-hot encoding of Y with shape (classes, m),
            or None on failure.
    """
    try:
        m = Y.shape[0]
        oh_encoding = np.zeros((classes, m))

        for index, value in enumerate(Y):
            oh_encoding[value, index] = 1

        return oh_encoding

    except Exception:
        return None
