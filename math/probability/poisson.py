#!/usr/bin/env python3
"""
    A script that represents a poisson distribution.
"""


class Poisson:
    """ A class Poisson that represents a poisson distribution. """

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

    def pmf(self, k):
        """
            Calculates the value of the PMF for a given number of "successes".

            Args:
                k (int): the number of "successes".

            Returns:
                The PMF value for k.
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        exp_negative_lambtha = e ** (-self.lambtha)
        lambtha_power_k = self.lambtha ** k

        pmf_value = (lambtha_power_k * exp_negative_lambtha) / factorial

        return pmf_value

    def cdf(self, k):
        """
            Calculates the value of the CDF for a given number of "successes".

            Args:
                k (int): the number of "successes".

            Returns:
                The CDF value for k.
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
