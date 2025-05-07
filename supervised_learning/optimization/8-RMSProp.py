#!/usr/bin/env python3
"""
    A function that sets up the RMSProp optimization algorithm in TensorFlow.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
        Sets up the RMSProp optimization algorithm in TensorFlow.

        Args:
            alpha (float): the learning rate.
            beta2 (float): the RMSProp weight (Discounting factor).
            epsilon (float): a small number to avoid division by zero.

        Returns:
            The optimizer.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
        )

    return optimizer
