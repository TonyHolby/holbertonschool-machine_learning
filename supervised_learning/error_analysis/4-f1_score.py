#!/usr/bin/env python3
"""
    A function that calculates the F1 score of a confusion matrix.
"""
import numpy as np


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
    actual_classes = confusion.shape[0]
    predictions = confusion.shape[1]
    precision = np.zeros(predictions)
    recall = np.zeros(predictions)
    classes_f1 = np.zeros(predictions)

    for real_class in range(actual_classes):
        true_positives = confusion[real_class, real_class]
        sum_of_real_classes = np.sum(confusion[real_class, :])
        recall[real_class] = true_positives / sum_of_real_classes

    for predicted_class in range(predictions):
        true_positives = confusion[predicted_class, predicted_class]
        sum_of_predicted_classes = np.sum(confusion[:, predicted_class])
        precision[
            predicted_class] = true_positives / sum_of_predicted_classes

    classes_f1 = (2 * precision * recall) / (precision + recall)

    return classes_f1
