#!/usr/bin/env python3
"""
    A script that defines a class named DeepNeuralNetwork that defines
    a deep neural network performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
        A class that defines a deep neural network performing
        binary classification.
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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
