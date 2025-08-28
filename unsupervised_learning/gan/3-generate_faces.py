#!/usr/bin/env python3
"""
    A function that builds and returns a convolutional generator and
    discriminator using tanh activation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolutional_GenDiscr():
    """
        Builds and returns a convolutional generator and discriminator.

        Returns:
            The concatenated output of the generator and discriminator.
    """

    def generator():
        """
            Builds the generator network using tanh activation.

            Returns:
                generator (keras.Model): the generator model.
        """
        inputs = keras.Input(shape=(16,))
        x = layers.Dense(8 * 8 * 64, activation="tanh")(inputs)
        x = layers.Reshape((8, 8, 64))(x)
        x = layers.Conv2DTranspose(32, kernel_size=3, strides=2,
                                   padding="same", activation="tanh")(x)
        x = layers.Conv2DTranspose(16, kernel_size=3, strides=1,
                                   padding="same", activation="tanh")(x)

        outputs = layers.Conv2D(1, kernel_size=3, strides=1,
                                padding="same", activation="tanh")(x)

        return keras.Model(inputs, outputs, name="generator")

    def discriminator():
        """
            Builds the discriminator network using tanh activation.

            Returns:
                discriminator (keras.Model): the discriminator model.
        """
        inputs = keras.Input(shape=(16, 16, 1))
        x = layers.Conv2D(32, kernel_size=3, strides=2,
                          padding="same", activation="tanh")(inputs)
        x = layers.Conv2D(64, kernel_size=3, strides=2,
                          padding="same", activation="tanh")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="tanh")(x)
        outputs = layers.Dense(1, activation="tanh")(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return generator(), discriminator()
