#!/usr/bin/env python3
"""
    A function that creates mini-batches to be used for training a neural
    network using mini-batch gradient descent.
"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
        Creates mini-batches to be used for training a neural network
        using mini-batch gradient descent.

        Args:
            X (np.ndarray): a numpy.ndarray of shape (m, nx) representing
                input data :
                    - m is the number of data points.
                    - nx is the number of features in X.
            Y (np.ndarray): a numpy.ndarray of shape (m, ny) representing
                the labels :
                    - m is the same number of data points as in X.
                    - ny is the number of classes for classification tasks.
            batch_size (int): the number of data points in a batch.

        Returns:
            A list of mini-batches containing tuples (X_batch, Y_batch).
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches_list = []

    for mini_batch_index in range(0, m, batch_size):
        X_batch = X_shuffled[mini_batch_index:mini_batch_index + batch_size]
        Y_batch = Y_shuffled[mini_batch_index:mini_batch_index + batch_size]
        mini_batches_list.append((X_batch, Y_batch))

    return mini_batches_list
