#!/usr/bin/env python3
"""
    A script that represents a normal distribution.
"""


class Normal:
    """ A class Normal that represents a normal distribution. """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            Initializes a normal distribution.

            Args:
                data (list): a list of the data to be used to estimate the
                distribution.
                mean (float): the mean of the distribution.
                stddev (float): the standard deviation of the distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            sum_of_squares = 0
            for x in data:
                sum_of_squares += (x - self.mean) ** 2
            variance = sum_of_squares / len(data)
            self.stddev = float(variance ** 0.5)
