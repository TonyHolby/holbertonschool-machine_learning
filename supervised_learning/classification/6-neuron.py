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
        """ Getter for the weights vector W of the neuron """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias b of the neuron """
        return self.__b

    @property
    def A(self):
        """ Getter for the activated output A of the neuron """
        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron.

            Arg:
                X (np.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.

            Returns:
                The private attribute __A.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

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
        cost_function = (1 / m) * (np.sum(loss_function))

        return cost_function

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                contains the correct labels for the input data.

            Returns:
                The neuron's prediction and the cost of the network.
        """
        A = self.forward_prop(X)
        neuron_prediction = np.where(A >= 0.5, 1, 0)
        network_cost = self.cost(Y, A)

        return neuron_prediction, network_cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
            and updates the private attributes __W and __b.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.
                A (np.ndarray): A numpy ndarray of shape (1, m) containing
                    the activated output of the neuron for each example.
                alpha (float): The learning rate.
        """
        m = Y.shape[1]
        partial_derivative_W = (1 / m) * np.dot((A - Y), X.T)
        partial_derivative_b = (1 / m) * np.sum(A - Y)
        self.__W = self.__W - alpha * partial_derivative_W
        self.__b = self.__b - alpha * partial_derivative_b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the neuron.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.
                alpha (float): The learning rate.
                iterations (int): The number of iterations to train over.

            Returns:
                The evaluation of the training data after iterations of
                training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for index in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
