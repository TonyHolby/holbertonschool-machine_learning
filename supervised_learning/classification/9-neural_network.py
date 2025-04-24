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
            Initializes a neural network with private instance attributes.

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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
            Getter for the weights vector W1 for the hidden layer
            of the neural network
        """
        return self.__W1

    @property
    def b1(self):
        """
            Getter for the bias b1 for the hidden layer
            of the neural network
        """
        return self.__b1

    @property
    def A1(self):
        """
            Getter for the activated output A1 for the hidden layer
            of the neural network
        """
        return self.__A1

    @property
    def W2(self):
        """ Getter for the weights vector W2 for the output neuron """
        return self.__W2

    @property
    def b2(self):
        """ Getter for the bias b2 for the output neuron """
        return self.__b2

    @property
    def A2(self):
        """
            Getter for the activated output A2 for the output neuron
            (prediction)
        """
        return self.__A2
