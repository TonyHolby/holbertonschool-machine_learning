#!/usr/bin/env python3
"""
    A function that builds a neural network with the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Builds a neural network with the Keras library.

        Arg:
            nx (int): the number of input features to the network.
            layers (list): a list containing the number of nodes in each layer
                of the network.
            activations (list): a list containing the activation functions used
                for each layer of the network.
            lambtha (float): the L2 regularization parameter.
            keep_prob (float): the probability that a node will be kept for
                dropout.

        Returns:
            The keras model.
    """
    model = K.Sequential()
    l2_regularization = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=l2_regularization,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=l2_regularization))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
