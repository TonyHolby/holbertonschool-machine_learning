#!/usr/bin/env python3
"""
    A script that defines a decision tree,
    determines the max depth of the tree,
    counts the number of nodes in the tree,
    displays the tree representation,
    returns the list of all leaves in the decision tree,
    computes the Node.lower and Node.upper bounds for each node,
    defines a function to check if x values are within the node's bounds,
    and computes the prediction functions update_predict() and pred().
"""
import numpy as np


class Node:
    """ A class named Node that updates the class Node """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
            Initializes a node.

            Args:
                feature (int): The index of the feature to split on.
                threshold (float): The threshold value to compare the
                feature against.
                left_child (Node or Leaf): Child node for values < threshold.
                right_child (Node or Leaf): Child node for values >= threshold.
                is_root (bool): True if this node is the root of the tree.
                depth (int): The depth of the node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Determines the depth of the node """
        if self.is_leaf:
            return 0

        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = 0

        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = 0

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
            Counts the number of nodes.

            Args:
                only_leaves (bool): Counts only if not leaf nodes.

            Returns:
                The number of nodes.
        """
        number_of_nodes = 0

        if not only_leaves or self.is_leaf:
            number_of_nodes += 1

        if self.left_child:
            number_of_nodes += self.left_child.count_nodes_below(only_leaves)

        if self.right_child:
            number_of_nodes += self.right_child.count_nodes_below(only_leaves)

        return number_of_nodes

    def __str__(self):
        """ Displays the tree as a string """
        def left_child_add_prefix(self, text):
            """
                Build the left children of the tree.

                Arg:
                    text (str): The description of the left children.

                Returns:
                    A new text to display the left part of the tree.
            """
            lines = text.split("\n")
            new_text = "    +--" + lines[0] + "\n"
            for x in lines[1:]:
                new_text += ("    |  " + x) + "\n"
            return (new_text)

        def right_child_add_prefix(self, text):
            """
                Build the right children of the tree.

                Arg:
                    text (str): The description of the right children.

                Returns:
                    A new text to display the right part of the tree.
            """
            lines = text.split("\n")
            new_text = "    +--" + lines[0] + "\n"
            for x in lines[1:]:
                new_text += ("       " + x) + "\n"
            return (new_text)

        if self.is_root:
            build_tree = \
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            build_tree = f"-> node [feature=\
{self.feature}, threshold={self.threshold}]\n"

        if self.left_child:
            build_tree += left_child_add_prefix(self, str(self.left_child))

        if self.right_child:
            build_tree += right_child_add_prefix(self, str(self.right_child))

        display_tree = build_tree.strip()

        return display_tree

    def get_leaves_below(self):
        """ Returns the list of all leaves in the decision tree """
        leaves_list = []
        if self.left_child:
            leaves_list.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves_list.extend(self.right_child.get_leaves_below())

        return leaves_list

    def update_bounds_below(self):
        """
            Computes, for each node, two dictionaries stored as
            attributes Node.lower and Node.upper
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                feature = self.feature
                threshold = self.threshold

                if child == self.right_child:
                    previous_upper = child.upper.get(feature, np.inf)
                    child.upper[feature] = min(previous_upper, threshold)
                else:
                    previous_lower = child.lower.get(feature, -np.inf)
                    child.lower[feature] = max(previous_lower, threshold)

                child.update_bounds_below()

    def update_indicator(self):
        """
            Defines an indicator function for a given node
            and checks if the x values are within the bounds
            self.lower and self.upper
        """
        def is_large_enough(x):
            """
                Checks if the feature values in x are greater
                than the node's lower bounds.

                Arg:
                    x (np.array): A 2D NumPy array
                    of shape (n_individuals, n_features).

                Returns:
                    A 1D boolean array,
                    of size equals to the number
                    of individuals (n_individuals),
                    containing boolean values.
            """
            return np.all(
                np.array([x[:, key] > self.lower[key]
                          for key in self.lower]).T, axis=1)

        def is_small_enough(x):
            """
                Checks if the feature values in x are less
                than or equal to the node's upper bounds.

                Arg:
                    x (np.ndarray): A 2D NumPy array
                    of shape (n_individuals, n_features).

                Returns:
                    A 1D boolean array,
                    of size equals to the number
                    of individuals (n_individuals),
                    containing boolean values.
            """
            return np.all(
                np.array([x[:, key] <= self.upper[key]
                          for key in self.upper]).T, axis=1)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

        return self.indicator

    def pred(self, x):
        """ Returns the prediction stored in a given node """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
        A class named Leaf that returns the depth
        and the number of leaves of a node
    """

    def __init__(self, value, depth=None):
        """ Initializes a leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns the depth of leaves """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Returns the number of leaves """
        return 1

    def __str__(self):
        """ Build the leaves description for the tree display """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """ A method that returns a leaf """
        return [self]

    def update_bounds_below(self):
        """ pass """
        pass

    def pred(self, x):
        """ Returns a prediciton stored in a leaf """
        return self.value


class Decision_Tree():
    """
        A class named Decision_tree that returns
        the max depth of the decision tree
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initializes a tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Returns the max depth of the decision tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns the number of nodes (without the leaves) """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ Returns the root description for the tree display """
        return self.root.__str__()

    def get_leaves(self):
        """ Returns all leaves in the decision tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Updates the decision tree bounds """
        self.root.update_bounds_below()

    def update_predict(self):
        """ Computes the prediction function """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: sum(leaf.indicator(A) * leaf.value
                                     for leaf in leaves)

    def pred(self, x):
        """
            Predicts the class of an individual.

            Arg:
                x (np.ndarray): A 1D numpy array of feature values.

            Returns:
                The predicted class of an individual.
        """
        return self.root.pred(x)
