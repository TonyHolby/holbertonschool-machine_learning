#!/usr/bin/env python3
"""
    A function that calculates the precision for each class in a confusion
    matrix.
"""
import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): a confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels:
                classes is the number of classes.

        Returns:
            A numpy.ndarray of shape (classes,) containing the precision
            of each class.
    """
    predictions = confusion.shape[1]
    classes_precision = np.zeros(predictions)

    for predicted_class in range(predictions):
        true_positives = confusion[predicted_class, predicted_class]
        sum_of_predicted_classes = np.sum(confusion[:, predicted_class])
        classes_precision[
            predicted_class] = true_positives / sum_of_predicted_classes

    return classes_precision
