#!/usr/bin/env python3
"""
    A function that builds a neural network with the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Builds a neural network with the Keras library using the Input class.

        Args:
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
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
