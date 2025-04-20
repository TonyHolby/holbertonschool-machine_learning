#!/usr/bin/env python3
"""
    A script that creates a new class Isolation_Random_Forest.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
            Initializes an Isolation Random Forest.

            Args:
                n_trees (int): Number of trees in the forest.
                max_depth (int): Maximum depth allowed for the tree.
                min_pop (int): Minimum population required to split a node.
                seed (int): Seed for random number generator.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
            Predicts the anomaly scores for the given dataset using
            the trained trees.

            Arg:
                explanatory (np.ndarray): The dataset to evaluate.

            Returns:
                The mean prediction across all trees in the forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])

        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
            Trains the isolation random forest on the provided dataset.

            Args:
                explanatory (np.ndarray): A 2D array of shape.
                n_trees (int): The number of trees in the forest.
                verbose (0 or 1): If set to 0, does not print training
                statistics after fitting.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
            Returns the indices of the n_suspects rows in explanatory that
            have the smallest anomaly scores (most anomalous points) and
            their corresponding scores.

            Args:
                explanatory (np.ndarray): The dataset to evaluate.
                n_suspects (int): The number of suspects to return.

            Returns:
                A tuple (suspects_values, anomaly_scores[outlier_indices])
        """
        anomaly_scores = self.predict(explanatory)
        n_suspects = min(n_suspects, len(explanatory))
        outlier_indices = np.argsort(anomaly_scores)[:n_suspects]
        suspects_values = explanatory[outlier_indices]

        return suspects_values, anomaly_scores[outlier_indices]
