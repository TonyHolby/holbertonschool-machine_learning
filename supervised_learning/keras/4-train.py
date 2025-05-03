#!/usr/bin/env python3
"""
    A function that trains a model using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
        Trains a model using mini-batch gradient descent.

        Args:
            network (tensorflow.keras.Model): the model to optimize.
            data (np.ndarray): a numpy.ndarray of shape (m, nx) containing the
                input data.
            labels (np.ndarray): a one-hot numpy.ndarray of shape (m, classes)
                containing the labels of data.
            batch_size (int): the size of the batch used for mini-batch
                gradient descent.
            epochs (int): the number of passes through data for mini-batch
                gradient descent.
            verbose (bool): a boolean that determines if output should be
                printed during training.
            shuffle (bool): a boolean that determines whether to shuffle the
                batches every epoch.

        Returns:
            The History object generated after training the model.
    """
    train_the_model = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return train_the_model
