#!/usr/bin/env python3
"""
    A function that sets up the Adam optimization algorithm in TensorFlow.
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
        Sets up the Adam optimization algorithm in TensorFlow.

        Args:
            alpha (float): the learning rate.
            beta1 (float): the weight used for the first moment
            beta2 (float): the weight used for the second moment.
            epsilon (float): a small number to avoid division by zero.

        Returns:
            The optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
        )

    return optimizer
