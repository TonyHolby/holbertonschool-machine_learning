#!/usr/bin/env python3
"""
    A function that creates a layer of a neural network using dropout.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
        Creates a layer of a neural network using dropout.

        Args:
            prev i(tensor): a tensor containing the output of the previous
            layer.
            n (int): the number of nodes the new layer should contain.
            activation (np.ndarray): the activation function for the new layer.
            keep_prob (float): the probability that a node will be kept.
            training (bool): a boolean indicating whether the model is in
            training mode.

        Returns:
            The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    output = dense(prev)

    if training:
        output = tf.nn.dropout(output, rate=1 - keep_prob)

    return output
