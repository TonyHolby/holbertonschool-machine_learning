#!/usr/bin/env python3
"""
    A function that builds a transition layer as described in Densely Connected
    Convolutional Networks.
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
        Builds a transition layer as described in Densely Connected
        Convolutional Networks.

        Args:
            X is the output from the previous layer.
            nb_filters is an integer representing the number of filters in X.
            compression is the compression factor for the transition layer.

        Returns:
            The output of the transition layer and the number of filters
            within the output, respectively.
    """
    he_normal = K.initializers.he_normal(seed=0)

    norm = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(norm)

    compressed_filters = int(nb_filters * compression)

    conv = K.layers.Conv2D(filters=compressed_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=he_normal,
                           use_bias=False)(relu)

    pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                     strides=2,
                                     padding='same')(conv)

    return pool, compressed_filters
