#!/usr/bin/env python3
"""
    A script that defines a decision tree
    and determines the max depth of the tree
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
                depth (int): Depth of the node.
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


class Leaf(Node):
    """ A class named Leaf that returns the depth of the leaf """

    def __init__(self, value, depth=None):
        """ Initializes a leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns the depth of the leaf """
        return self.depth


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
