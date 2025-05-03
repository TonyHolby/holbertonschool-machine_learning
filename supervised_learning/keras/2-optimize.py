#!/usr/bin/env python3
"""
    A function that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics.
"""
from tensorflow.keras.optimizers import Adam


def optimize_model(network, alpha, beta1, beta2):
    """
        Sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics.

        Arg:
            network (tensorflow.keras.Model): the model to optimize.
            alpha (float): the learning rate.
            beta1 (float): the first Adam optimization parameter.
            beta2 (float): the second Adam optimization parameter.

        Returns:
            None.
    """
    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
