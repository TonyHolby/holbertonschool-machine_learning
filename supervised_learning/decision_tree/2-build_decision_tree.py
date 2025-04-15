#!/usr/bin/env python3
"""
    A script that defines a decision tree,
    determines the max depth of the tree
    and counts the number of nodes.
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
        def left_child_add_prefix(self, text):
            lines = text.split("\n")
            new_text = "    +--" + lines[0] + "\n"
            for x in lines[1:]:
                new_text += ("    |  " + x) + "\n"
            return (new_text)

        def right_child_add_prefix(self, text):
            lines = text.split("\n")
            new_text = "    +--" + lines[0] + "\n"
            for x in lines[1:]:
                new_text += ("    |  " + x) + "\n"
            return (new_text)

        if self.is_root:
            build_tree = \
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            build_tree = \
                f"node [feature={self.feature}, threshold={self.threshold}]\n"

        if self.left_child:
            build_tree += left_child_add_prefix(self, str(self.left_child))

        if self.right_child:
            build_tree += right_child_add_prefix(self, str(self.right_child))

        display_tree = build_tree.strip()

        return display_tree


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
        return (f"-> leaf [value={self.value}]")


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
        """ Returns the number of nodes """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return self.root.__str__()
