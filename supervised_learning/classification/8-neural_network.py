#!/usr/bin/env python3
"""
    A script that defines a class named NeuralNetwork that defines a
    neural network with one hidden layer performing binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
        A class that defines a neural network with one hidden layer
        performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
            Initializes a neural network.

            Arg:
                nx (int): The number of input features.
                nodes (int): The number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
