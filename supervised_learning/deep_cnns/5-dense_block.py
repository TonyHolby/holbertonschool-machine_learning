#!/usr/bin/env python3
"""
    A function that builds a dense block as described in Densely Connected
    Convolutional Networks.
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
        Builds a dense block as described in Densely Connected Convolutional
        Networks.

        Args:
            X is the output from the previous layer.
            nb_filters is an integer representing the number of filters in X.
            growth_rate is the growth rate for the dense block.
            layers is the number of layers in the dense block.

        Returns:
            The concatenated output of each layer within the Dense Block and
            the number of filters within the concatenated outputs.
    """
    he_normal = K.initializers.he_normal(seed=0)

    for i in range(layers):
        X1 = K.layers.BatchNormalization(axis=-1)(X)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(4 * growth_rate,
                             (1, 1),
                             padding='same',
                             kernel_initializer=he_normal)(X1)

        X1 = K.layers.BatchNormalization(axis=-1)(X1)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(growth_rate,
                             (3, 3),
                             padding='same',
                             kernel_initializer=he_normal)(X1)

        X = K.layers.Concatenate(axis=-1)([X, X1])

        nb_filters += growth_rate

    return X, nb_filters
