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

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.
                A (np.ndarray): A numpy ndarray of shape (1, m) containing
                    the activated output of the neuron for each example.

            Returns:
                The cost of the model.
        """
        loss_function = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        m = Y.shape[1]
        cost_function = (1 / m) * np.sum(loss_function)

        return cost_function

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.

            Returns:
                The neuron's prediction and the cost of the network.
        """
        neural_network_output, _ = self.forward_prop(X)

        neuron_prediction = np.where(neural_network_output >= 0.5, 1, 0)
        network_cost = self.cost(Y, neural_network_output)

        return neuron_prediction, network_cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
            and updates the private attribute __weights.

            Args:
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.
                cache (dict): a dictionary containing all the intermediary
                    values of the network.
                alpha (float): The learning rate.
        """
        m = Y.shape[1]

        dZ = self.__cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            dZl = self.__cache['A' + str(i)] - Y
            dWl = (1 / m) * np.matmul(dZl, self.__cache['A' + str(i-1)].T)
            dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)

            self.__weights['W' + str(i)] = self.__weights[
                'W' + str(i)] - alpha * dWl
            self.__weights['b' + str(i)] = self.__weights[
                'b' + str(i)] - alpha * dbl
