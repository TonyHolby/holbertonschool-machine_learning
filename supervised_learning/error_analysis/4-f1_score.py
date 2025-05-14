#!/usr/bin/env python3
"""
    A function that calculates the F1 score of a confusion matrix.
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        Calculates the F1 score of a confusion matrix.

        Args:
            confusion (np.ndarray): a confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels:
                classes is the number of classes.

        Returns:
            A numpy.ndarray of shape (classes,) containing the F1 score
            of each class.
    """
    accuracy = precision(confusion)
    recall = sensitivity(confusion)
    classes_f1 = (2 * accuracy * recall) / (accuracy + recall)

    return classes_f1
