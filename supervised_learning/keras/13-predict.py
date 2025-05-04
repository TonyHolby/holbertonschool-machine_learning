#!/usr/bin/env python3
"""
    A function that makes a prediction using a neural network.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        Makes a prediction using a neural network.

        Args:
            network (tensorflow.keras.Model): the network model to make the
                prediction with.
            data (np.ndarray): the input data to make the prediction with.
            verbose (bool): a boolean that determines if output should
                be printed during the prediction process.

        Returns:
            The prediction for the data.
    """
    prediction = network.predict(x=data, verbose=verbose)

    return prediction
