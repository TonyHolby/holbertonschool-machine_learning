#!/usr/bin/env python3
"""
    A function that calculates the positional encoding for a transformer.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
        Calculates the positional encoding for a transformer.

        Args:
            max_seq_len (int): an integer representing the maximum sequence
                length.
            dm (int): the model depth.

        Returns:
            a numpy.ndarray of shape (max_seq_len, dm) containing the
            positional encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(dm))
    angle_rads = positions * angle_rates
    pos_encoding = np.zeros_like(angle_rads)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pos_encoding
