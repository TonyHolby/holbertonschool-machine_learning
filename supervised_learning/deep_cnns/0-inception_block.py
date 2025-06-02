#!/usr/bin/env python3
"""
    A function that builds an inception block as described in Going Deeper
    with Convolutions (2014).
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
        Builds an inception block as described in Going Deeper with
        Convolutions (2014).

        Args:
            A_prev (np.ndarray): the output from the previous layer.
            filters (tuple or list): a tuple or list containing F1, F3R, F3,
            F5R, F5, FPP, respectively:
                F1 is the number of filters in the 1x1 convolution.
                F3R is the number of filters in the 1x1 convolution before the
                3x3 convolution.
                F3 is the number of filters in the 3x3 convolution.
                F5R is the number of filters in the 1x1 convolution before the
                5x5 convolution.
                F5 is the number of filters in the 5x5 convolution.
                FPP is the number of filters in the 1x1 convolution after the
                max pooling.

        Returns:
            The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(F1,
                               (1, 1),
                               padding='same',
                               activation='relu')(A_prev)

    conv_1x1_before_3x3 = K.layers.Conv2D(F3R,
                                          (1, 1),
                                          padding='same',
                                          activation='relu')(A_prev)

    conv_3x3 = K.layers.Conv2D(F3,
                               (3, 3),
                               padding='same',
                               activation='relu')(conv_1x1_before_3x3)

    conv_1x1_before_5x5 = K.layers.Conv2D(F5R,
                                          (1, 1),
                                          padding='same',
                                          activation='relu')(A_prev)

    conv_5x5 = K.layers.Conv2D(F5,
                               (5, 5),
                               padding='same',
                               activation='relu')(conv_1x1_before_5x5)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(A_prev)

    conv_1x1_after_max_pool = K.layers.Conv2D(FPP,
                                              (1, 1),
                                              padding='same',
                                              activation='relu')(max_pool)

    concatened_output = K.layers.Concatenate(axis=-1)(
        [conv_1x1, conv_3x3, conv_5x5, conv_1x1_after_max_pool])

    return concatened_output
