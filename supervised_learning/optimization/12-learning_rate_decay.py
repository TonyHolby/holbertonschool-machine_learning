#!/usr/bin/env python3
"""
    A function that creates a learning rate decay operation in tensorflow
    using inverse time decay.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
        Creates a learning rate decay operation in tensorflow using inverse
        time decay.

        Args:
            alpha (float): the original learning rate.
            decay_rate (float): the weight used to determine the rate at
                which alpha will decay.
            decay_step (int): the number of passes of gradient descent that
                should occur before alpha is decayed further.

        Returns:
            The learning rate decay operation.
    """
    new_alpha = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True,
        name='InverseTimeDecay'
        )

    return new_alpha
