#!/usr/bin/env python3
"""
    A script that represents an exponential distribution.
"""


class Exponential:
    """ A class Exponential that represents an exponential distribution. """

    def __init__(self, data=None, lambtha=1.):
        """
            Initializes an exponential distribution.

            Args:
                data (list): a list of the data to be used to estimate the
                distribution.
                lambtha (float): the expected number of occurences in a given
                time frame.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)
