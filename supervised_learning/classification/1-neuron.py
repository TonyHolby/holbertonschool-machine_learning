#!/usr/bin/env python3
"""
    A script that defines a class named Neuron that defines a single
    neuron performing binary classification.
"""
import numpy as np


class Neuron:
    """
        A class that defines a single neuron performing binary classification.
    """
    def __init__(self, nx):
        """
            Initializes a neuron with private instance attributes.

            Arg:
                nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
