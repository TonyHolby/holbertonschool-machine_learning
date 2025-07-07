#!/usr/bin/env python3
"""
    A script that represents a poisson distribution.
"""


class Poisson:
    """ a class Poisson that represents a poisson distribution. """

    def __init__(self, data=None, lambtha=1.):
        """
            Initializes a poisson distribution.

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
            sum = 0
            number_of_value = 0
            for value in data:
                sum += value
                number_of_value += 1
            self.lambtha = float(sum / number_of_value)
