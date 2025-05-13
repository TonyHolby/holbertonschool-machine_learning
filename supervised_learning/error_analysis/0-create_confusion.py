#!/usr/bin/env python3
"""
    A function that creates a confusion matrix.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Creates a confusion matrix.

        Args:
            labels (np.ndarray): a one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels for each data point :
                    m is the number of data points.
                    classes is the number of classes.
            logits (np.ndarray): a one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels.

        Returns:
            A confusion numpy.ndarray of shape (classes, classes) with row
            indices representing the correct labels and column indices
            representing the predicted labels.
    """
    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes), dtype=float)

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for true_index, pred_index in zip(true_labels, pred_labels):
        confusion_matrix[true_index, pred_index] += 1

    return confusion_matrix
