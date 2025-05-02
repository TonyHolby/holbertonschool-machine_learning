#!/usr/bin/env python3
"""
    A script that defines a class named DeepNeuralNetwork that defines
    a deep neural network performing multiclass classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
        A class that defines a deep neural network performing
        multiclass classification with private instance attributes.
    """
    def __init__(self, nx, layers, activation='sig'):
        """
            Initializes a deep neural network.

            Args:
                nx (int): The number of input features.
                layers (list): A list representing the number of
                    nodes in each layer of the network.
                activation (str): Represents the type of activation
                    function used in the hidden layers :
                    - sig : represents a sigmoid activation
                    - tanh : represents a tanh activation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda i: isinstance(i, int) and i > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        if activation != "sig" and activation != "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            layer_input = nx if i == 0 else layers[i - 1]
            if self.__activation == 'tanh':
                weight_init = np.sqrt(1 / layer_input)
            else:
                weight_init = np.sqrt(2 / layer_input)

            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], layer_input) * weight_init)
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

    @property
    def activation(self):
        """
            Getter for the value of activation
        """
        return self.__activation

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Arg:
                X (np.ndarray): A numpy.ndarray with shape (nx, m)
                    that contains the input data.

            Returns:
                The output of the neural network (softmax probabilities)
                and the cache, respectively.
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            Z = np.matmul(
                self.__weights['W' + str(i)], self.__cache['A' + str(i - 1)]
                ) + self.__weights['b' + str(i)]
            if i != self.__L:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    A = (2 / (1 + np.exp(-2 * Z))) - 1
            else:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache['A' + str(i)] = A

        Y_hat = self.__cache['A' + str(self.__L)]

        return Y_hat, self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model for multiclass classification
            using softmax activation.

            Args:
                Y (np.ndarray): A one_hot numpy ndarray of shape (classes, m)
                    that contains the correct labels.
                A (np.ndarray): A one_hot numpy ndarray of shape (classes, m)
                    containing the predicted probabilities from the softmax
                    output.

            Returns:
                The cost of the model.
        """
        loss_function = -np.sum(Y * np.log(A + 1e-8))
        m = Y.shape[1]
        cost_function = (1 / m) * loss_function

        return cost_function

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A one_hot numpy ndarray of shape (classes, m)
                    that contains the correct labels.

            Returns:
                The neuron's prediction and the cost of the network.
        """
        Y_hat, _ = self.forward_prop(X)
        network_cost = self.cost(Y, Y_hat)

        predicted_classes = np.argmax(Y_hat, axis=0)
        m = Y_hat.shape[1]
        classes = Y_hat.shape[0]
        predictions = np.zeros((classes, m))
        predictions[predicted_classes, np.arange(m)] = 1

        return predictions, network_cost

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
        copy_of_weights = self.__weights.copy()

        dZ = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W_current = copy_of_weights['W' + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

            if i > 1:
                dA_prev = np.matmul(W_current.T, dZ)
                A_prev = cache['A' + str(i - 1)]

                if self.__activation == 'sig':
                    dZ = dA_prev * A_prev * (1 - A_prev)
                elif self.__activation == 'tanh':
                    dZ = dA_prev * (1 - A_prev ** 2)

    def train(self, X, Y, iterations, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Trains the deep neural network by updating the private
            attributes __weights and __cache.

            Args:
                X (np.ndarray): A numpy ndarray of shape (nx, m) that
                    contains input data.
                Y (np.ndarray): A one_hot numpy ndarray of shape (classes, m)
                    that contains the correct labels for the input data.
                alpha (float): The learning rate.
                iterations (int): The number of iterations to train over.
                verbose (boolean): A boolean that defines whether or not
                    to print information about the training.
                graph (boolean): A boolean that defines whether or not
                    to graph information about the training once the
                    training has completed.

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

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step > iterations:
            step = iterations
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        cost_during_iteration = []
        step_counter = []

        for i in range(iterations + 1):
            activations, self.__cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, activations)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    cost_during_iteration.append(cost)
                    step_counter.append(i)

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot(step_counter, cost_during_iteration, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format.

            Args:
                filename (str): The file to which the object should be
                saved.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, 'wb') as file_to_save:
            pickle.dump(self, file_to_save)

    @staticmethod
    def load(filename):
        """
            Loads a pickled DeepNeuralNetwork object.

            Args:
                filename (str): The file from which the object should be
                loaded.

            Returns:
                The loaded object, or None if filename doesn't exist.
        """
        try:
            with open(filename, "rb") as file_to_load:
                model = pickle.load(file_to_load)

            return model

        except Exception:
            return None
