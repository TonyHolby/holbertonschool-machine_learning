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

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Arg:
                X (np.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.

            Returns:
                The private attributes __A1 and __A2.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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
            Evaluates the neural network's predictions.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.

            Returns:
                The neuron's prediction and the cost of the network.
        """
        _, A2 = self.forward_prop(X)
        neuron_prediction = np.where(A2 >= 0.5, 1, 0)
        network_cost = self.cost(Y, A2)

        return neuron_prediction, network_cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
            and updates the private attributes _W1, __b1, __W2, and __b2.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A numpy ndarray of shape (1, m) that
                    contains the correct labels for the input data.
                A1 (np.ndarray): The output of the hidden layer.
                A2 (np.ndarray): The predicted output.
                alpha (float): The learning rate.
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
