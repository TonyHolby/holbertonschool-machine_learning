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
        if not all(map(lambda element: isinstance(
            element, int) and element > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L):
            self.weights['W' + str(i)] = np.random.randn(
                layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.weights['b' + str(i)] = np.zeros((layers[i], 1))

        self.weights['W' + str(self.L)] = np.random.randn(
            layers[self.L-1], layers[self.L-2]) * np.sqrt(2 / layers[self.L-2])
        self.weights['b' + str(self.L)] = np.zeros((layers[self.L-1], 1))
