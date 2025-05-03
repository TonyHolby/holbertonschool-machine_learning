#!/usr/bin/env python3
"""
    A script that saves and loads a model's weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
        Saves a model's weights.

        Args:
            network (tensorflow.keras.Model): the model whose weights should
                be saved.
            filename (str): the path of the file that the weights should be
                saved to.
            save_format (str): the format in which the weights should be saved.

        Returns:
            None.
    """
    if save_format.endswith('.keras'):
        save_format.replace('.keras', '.weights.h5')

    network.save_weights(filename, save_format)


def load_weights(network, filename):
    """
        Loads a model's weights.

        Args:
            network (tensorflow.keras.Model): the model whose weights should
                be saved.
            filename (str): the path of the file that the weights should be
                saved to.

        Returns:
            None.
    """
    network.load_weights(filename)
