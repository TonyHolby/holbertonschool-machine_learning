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
            Initializes a neuron.

            Arg:
                nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
