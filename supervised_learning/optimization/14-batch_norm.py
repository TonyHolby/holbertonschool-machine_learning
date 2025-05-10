#!/usr/bin/env python3
"""
    A function that creates a batch normalization layer for a neural network
    in tensorflow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Creates a batch normalization layer for a neural network in tensorflow.

        Args:
            prev (tf.Tensor): the activated output of the previous layer.
            n (int): the number of nodes in the layer to be created.
            activation (function): the activation function that should be used
                on the output of the layer.

        Returns:
            A tensor of the activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer
    )(prev)

    mean, variance = tf.nn.moments(dense, axes=[0], keepdims=True)
    gamma = tf.Variable(tf.ones([1, n]), trainable=True)
    beta = tf.Variable(tf.zeros([1, n]), trainable=True)
    epsilon = 1e-7
    normalized_output = (dense - mean) / tf.sqrt(variance + epsilon)
    scaled_output = gamma * normalized_output + beta
    activated_output = activation(scaled_output)

    return activated_output
