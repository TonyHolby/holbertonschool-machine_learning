#!/usr/bin/env python3
"""
    A script that defines a class named DeepNeuralNetwork that defines
    a deep neural network performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
        A class that defines a deep neural network performing
        binary classification with private instance attributes.
    """
    def __init__(self, nx, layers):
        """
            Initializes a deep neural network.

            Args:
                nx (int): The number of input features.
                layers (list): A list representing the number of
                    nodes in each layer of the network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda i: isinstance(i, int) and i > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
            Getter for the number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
            Getter for the dictionary to hold all intermediary
            values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
            Getter for the dictionary to hold all weights and
            biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Arg:
                X (np.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.

            Returns:
                The output of the neural network and the cache, respectively.
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(i)]
            bl = self.__weights['b' + str(i)]
            Al_prev = self.__cache['A' + str(i-1)]

            Zl = np.dot(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))

            self.__cache['A' + str(i)] = Al

        neural_network_output = self.__cache['A' + str(self.__L)]

        return neural_network_output, self.__cache
