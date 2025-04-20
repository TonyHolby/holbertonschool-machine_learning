#!/usr/bin/env python3
"""
    A script that creates a new class Random_Forest.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
        A class named Random Forest that :
        - build a large list of decision trees,
        - predicts the class of each tree in the forest,
        - trains the random forest model,
        - computes the accuracy of the model.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
            Initializes the forest with hyperparameters.

            Args:
                n_trees (int): Number of decision trees in the forest.
                max_depth (int): Maximum depth allowed for each tree.
                min_pop (int): Minimum number of samples required to continue
                splitting a node.
                seed (int): Random seed for reproducibility.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
            Generate predictions for each tree in the forest.

            Arg:
                explanatory (np.ndarray): The feature matrix of input data.

            Returns:
                The predicted class labels for each input sample.
        """
        predictions = np.array([tree(
            explanatory) for tree in self.numpy_preds])

        predictions = predictions.T

        def calculate_mode(predictions):
            """
                Calculate the mode (most frequent) prediction for each example.

                Arg:
                    predictions (np.ndarray): A 2D array of shape
                    (n_samples, n_trees), where each row contains predictions
                    from all trees for a single input sample.

                Returns:
                    An array of shape (n_samples,) containing the
                    majority-voted class for each sample.
            """
            majority_votes = []
            for row in predictions:
                values, counts = np.unique(row, return_counts=True)
                majority = values[np.argmax(counts)]
                majority_votes.append(majority)
            return np.array(majority_votes)

        majority_votes = calculate_mode(predictions)

        return majority_votes

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
            Trains a forest composed of multiple decision trees.

            Args:
                explanatory (np.ndarray): Feature matrix.
                target (np.ndarray): Target labels.
                n_trees (int): Number of trees to train.
                verbose (0 or 1): If set to 1, prints training statistics
                after fitting. Default is 0.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(
        self.explanatory,self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
            Computes the accuracy of the random forest on the given test data.

            Arg:
                test_explanatory (np.ndarray): A 2D array of shape
                containing the test input features.
                test_target (np.ndarray): A 1D array of shape (n_samples,)
                containing the true test labels.

            Returns:
                The accuracy of the model (between 0 and 1).
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target))/test_target.size
