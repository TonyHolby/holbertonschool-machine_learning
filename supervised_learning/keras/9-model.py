#!/usr/bin/env python3
"""
    A script that saves and load an entire model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
        Saves an entire model.

        Args:
            network (tensorflow.keras.Model): the model to save.
            filename (str): the path of the file that the model should be
            saved to.

        Returns:
            None.
    """
    network.save(filename)


def load_model(filename):
    """
        Loads an entire model.

        Args:
            filename (str): the path of the file that the model should be
            saved to.

        Returns:
            The loaded model.
    """
    model_loaded = K.models.load_model(filename)

    return model_loaded
