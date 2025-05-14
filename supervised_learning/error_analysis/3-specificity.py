#!/usr/bin/env python3
"""
    A function that calculates the specificity for each class in a confusion
    matrix.
"""
import numpy as np


def specificity(confusion):
    """
        Calculates the specificity for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): a confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels:
                classes is the number of classes.

        Returns:
            A numpy.ndarray of shape (classes,) containing the specificity
            of each class.
    """
    actual_classes = confusion.shape[0]
    all_classes = np.sum(confusion)
    classes_specificity = np.zeros(actual_classes)

    for real_class in range(actual_classes):
        true_positives = confusion[real_class, real_class]
        false_positives = np.sum(confusion[:, real_class]) - true_positives
        false_negatives = np.sum(confusion[real_class, :]) - true_positives
        true_negatives = \
            all_classes - true_positives - false_positives - false_negatives
        classes_specificity[
            real_class] = true_negatives / (true_negatives + false_positives)

    return classes_specificity
