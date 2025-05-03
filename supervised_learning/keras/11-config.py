#!/usr/bin/env python3
"""
    A script that saves and loads a model's configuration in JSON format.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
        Saves a model's configuration in JSON format.

        Args:
            network (tensorflow.keras.Model): the model whose configuration
                should be saved.
            filename (str): the path of the file that the configuration
                should be saved to.

        Returns:
            None.
    """
    config_json = network.to_json()
    with open(filename, 'w') as f:
        f.write(config_json)


def load_config(filename):
    """
        Loads a model with a specific configuration.

        Args:
            filename (str): the path of the file containing the model's
                configuration in JSON format.

        Returns:
            The loaded model.
    """
    with open(filename, 'r') as f:
        config_json = f.read()
    model = K.models.model_from_json(config_json)

    return model
