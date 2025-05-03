#!/usr/bin/env python3
"""
    A function that trains a model with learning rate decay,
    analyzes validation data,
    saves the best iteration of the model,
    using mini-batch gradient descent and
    early stopping.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
        Trains a model with learning rate decay, using mini-batch gradient
        descent, early stopping, analyzes validation data, saves the best
        iteration of the model.

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
            learning_rate_decay (bool): a boolean that indicates whether
                learning rate decay should be used.
            alpha (float): the initial learning rate.
            decay_rate (float): the decay rate.
            save_best (bool): a boolean indicating whether to save the model
                after each epoch if it is the best.
            filepath (str): the file path where the model should be saved.

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

    if learning_rate_decay and validation_data is not None:
        if hasattr(network.optimizer, 'learning_rate'):
            network.optimizer.learning_rate = alpha
        elif hasattr(network.optimizer, 'lr'):
            network.optimizer.lr = alpha

        lr_scheduler = K.callbacks.LearningRateScheduler(
            lambda epoch: alpha / (1 + decay_rate * epoch),
            verbose=1
        )
        callbacks.append(lr_scheduler)

    if save_best and validation_data is not None and filepath is not None:
        model_checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(model_checkpoint)

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
