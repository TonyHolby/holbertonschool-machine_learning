#!/usr/bin/env python3
"""
    A function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015).
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
        Builds an identity block as described in Deep Residual Learning
        for Image Recognition (2015).

        Args:
            A_prev (np.ndarray): the output from the previous layer.
            filters (tuple or list): a tuple or list containing F11, F3, F12,
            respectively:
                F11 is the number of filters in the first 1x1 convolution.
                F3 is the number of filters in the 3x3 convolution.
                F12 is the number of filters in the second 1x1 convolution.

        Returns:
            The activated output of the identity block.
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.HeNormal(seed=0)

    x = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(A_prev)

    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=he_normal)(x)

    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(x)

    x = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
