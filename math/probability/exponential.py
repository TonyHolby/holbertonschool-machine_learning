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

    def pdf(self, x):
        """
            Calculates the value of the PDF for a given time period.

            Args:
                x (float): the time period.

            Returns:
                The PDF value for x.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        pdf_value = self.lambtha * (e ** (-self.lambtha * x))

        return pdf_value

    def cdf(self, x):
        """
            Calculates the value of the CDF for a given time period.

            Args:
                k (float): the time period.

            Returns:
                The CDF value for x.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        cdf_value = 1 - (e ** (-self.lambtha * x))

        return cdf_value
