#!/usr/bin/env python3
"""
    A function that trains a model using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
        Trains a model using mini-batch gradient descent, early stopping
        and analyze validation data.

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
            validation_data (np.ndarray): the data to validate the model with,
                if not None.
            early_stopping (bool): a boolean that indicates whether early
                stopping should be used.
            patience (int): the patience used for early stopping.


        Returns:
            The History object generated after training the model.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True)
        callbacks.append(early_stop)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
