#!/usr/bin/env python3
"""
    A function that tests a neural network.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        Tests a neural network.

        Args:
            network (tensorflow.keras.Model): the network model to test.
            data (np.ndarray): the input data to test the model with.
            labels (np.ndarray): the correct one-hot labels of data.
            verbose (bool): a boolean that determines if output should be
                printed during the testing process.

        Returns:
            The loss and accuracy of the model with the testing data.
    """
    history = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )

    return history
