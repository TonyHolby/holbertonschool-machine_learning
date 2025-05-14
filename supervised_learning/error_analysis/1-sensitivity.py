#!/usr/bin/env python3
"""
    A function that calculates the sensitivity for each class in a confusion
    matrix.
"""
import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): a confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels:
                classes is the number of classes.

        Returns:
            A numpy.ndarray of shape (classes,) containing the sensitivity
            of each class.
    """
    actual_classes = confusion.shape[0]
    classes_sensitivity = np.zeros(actual_classes)

    for real_class in range(actual_classes):
        true_positives = confusion[real_class, real_class]
        sum_of_real_classes = np.sum(confusion[real_class, :])
        classes_sensitivity[real_class] = true_positives / sum_of_real_classes

    return classes_sensitivity
