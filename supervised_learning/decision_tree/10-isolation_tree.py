#!/usr/bin/env python3
"""
    A script that creates a new class Isolation_Random_Tree.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
        A class named Isolation_Random_Tree that defines an
        Isolation Random Tree
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
            Initializes an Isolation Random Tree.

            Args:
                max_depth (int): Maximum depth allowed for the tree.
                seed (int): Seed for random number generator.
                root (Node, optional): An existing root node.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ same as in Decision_Tree """
        return str(self.root)

    def depth(self):
        """ same as in Decision_Tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ same as in Decision_Tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """ same as in Decision_Tree """
        self.root.update_bounds_below()

    def get_leaves(self):
        """ same as in Decision_Tree """
        return self.root.get_leaves_below()

    def update_predict(self):
        """ Updates the bounds and the indicator function """
        self.root.update_bounds_below()
        self.root.update_indicator()

        def tree_predict(x):
            preds = []
            for i in range(x.shape[0]):
                preds.append(self.root.pred(x[i]))
            return np.array(preds)

        self.predict = tree_predict

    def np_extrema(self, arr):
        """
            Computes the minimum and maximum values of a numpy array.

            Arg:
                arr (np.ndarray): A 1D numpy array of numerical values.

            Returns:
                A tuple (min_value, max_value).
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
            Randomly selects a feature and a threshold to split the data at
            the given node.

            Arg:
                node (Node): The current node containing a sub_population of
                individuals.

            Returns:
                A tuple (index of the feature used for splitting,
                        threshold value used to split the feature).
        """
        sub_pop = node.sub_population
        x = self.explanatory[sub_pop]
        n_features = x.shape[1]

        while True:
            feature = self.rng.integers(0, n_features)
            col = x[:, feature]
            min_val, max_val = self.np_extrema(col)
            if max_val > min_val:
                break

        threshold = self.rng.uniform(min_val, max_val)
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
            Creates a leaf node for a given subset of the population.

            Args:
                node (Node): The parent node from which this leaf descends.
                sub_population (np.ndarray): A boolean array indicating which
                individuals belong to this leaf.

            Returns:
                A leaf node with its class value set to the most common target.
        """
        leaf_child = Leaf(value=node.depth + 1, depth=node.depth + 1)
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
            Creates a new node as a child of the given parent node.

            Args:
                node (Node): The parent node.
                sub_population (np.ndarray): A boolean array indicating which
                individuals belong to this node.

            Returns:
                A new node with updated depth and population.
        """
        return Node(is_root=False, depth=node.depth + 1, left_child=None,
                    right_child=None, feature=None, threshold=None)

    def fit_node(self, node):
        """
            Recursively builds the decision tree starting from the given node.

            Arg:
                node (Node): The current node to split.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        x = self.explanatory
        sub_pop = node.sub_population

        left_mask = (x[:, node.feature] > node.threshold) & sub_pop
        right_mask = (x[:, node.feature] <= node.threshold) & sub_pop

        left_population = left_mask
        right_population = right_mask

        is_left_leaf = (node.depth + 1 >= self.max_depth) or (np.sum(
            left_population) <= self.min_pop)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.sub_population = left_population
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 >= self.max_depth) or (np.sum(
            right_population) <= self.min_pop)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.sub_population = right_population
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
            Trains the decision tree on the provided dataset.

            Args:
                explanatory (np.ndarray): A 2D array of shape.
                target (np.ndarray): A 1D numpy array target of size number
                of individuals.
                verbose (0 or 1): If set to 1, prints training statistics
                after fitting. Default is 0.
        """
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
