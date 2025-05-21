#!/usr/bin/env python3
"""
    A function that calculates the cost of a neural network with L2
    regularization.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
        Calculates the cost of a neural network with L2 regularization.

        Args:
            cost (tf.tensor): a tensor containing the cost of the network
            without L2 regularization.
            model (tf.keras.Model): a Keras model that includes layers with L2
            regularization.

        Returns:
            A tensor containing the total cost for each layer of the network,
            accounting for L2 regularization.
    """
    l2_loss = tf.add_n(model.losses)
    total_cost = cost + l2_loss

    return total_cost
