#!/usr/bin/env python3
"""
    A function that converts a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
        Converts a label vector into a one-hot matrix.

        Args:
            labels (np.ndarray): A numpy ndarray of shape (m,) containing
                the labels for the input data.
            classes (int): The maximum number of classes found in labels.

        Returns:
            The one-hot matrix.
    """
    labels = K.backend.cast(K.backend.constant(labels), 'int32')

    if classes is None:
        classes = K.backend.eval(K.backend.max(labels)) + 1

    one_hot_matrix = K.backend.one_hot(labels, num_classes=classes)

    return K.backend.eval(one_hot_matrix)
