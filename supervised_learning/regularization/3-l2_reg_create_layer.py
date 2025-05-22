#!/usr/bin/env python3
"""
    A function that creates a neural network layer in tensorFlow that includes
    L2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a neural network layer in tensorFlow that includes L2
        regularization.

        Args:
            prev (tensor): a tensor containing the output of the previous layer
            n (int): the number of nodes the new layer should contain
            activation (np.ndarray): the activation function that should be
            used on the layer.
            lambtha (float): the L2 regularization parameter.

        Returns:
            The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    regularizer = tf.keras.regularizers.L2(lambtha)

    new_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    output = new_layer(prev)

    return output
