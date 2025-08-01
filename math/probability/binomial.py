#!/usr/bin/env python3
"""
    A script that represents a binomial distribution.
"""


class Binomial:
    """ A class Binomial that represents a binomial distribution. """

    def __init__(self, data=None, n=1, p=0.5):
        """
            Initializes a binomial distribution.

            Args:
                data (list): a list of the data to be used to estimate the
                distribution.
                n (int): the number of Bernoulli trials.
                p (float): the probability of a "success".
        """
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not isinstance(p, (int, float)) or not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            number_of_value = len(data)
            sum = 0
            for value in data:
                sum += value

            mean = sum / number_of_value

            sum_of_squares = 0
            for value in data:
                sum_of_squares += (value - mean) ** 2

            variance = sum_of_squares / number_of_value

            probability_of_success = 1 - (variance / mean)
            trials = round(mean / probability_of_success)
            probability_of_success = mean / trials

            self.n = trials
            self.p = probability_of_success

    def pmf(self, k):
        """
            Calculates the value of the PMF for a given number of "successes".

            Args:
                k (int or float): the number of "successes".

            Returns:
                The PMF value for k.
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0 or k > self.n:
            return 0

        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        n_choose_k = 1
        for i in range(1, (self.n - k) + 1):
            n_choose_k *= i

        combination = n_factorial // (k_factorial * n_choose_k)
        pmf_value = combination * (self.p ** k) * (
            (1 - self.p) ** (self.n - k))

        return pmf_value

    def cdf(self, k):
        """
            Calculates the value of the CDF for a given number of "successes".

            Args:
                k (int or float): the number of "successes".

            Returns:
                The CDF value for k.
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0 or k > self.n:
            return 0

        cdf_value = 0

        for x in range(0, k + 1):
            n_factorial = 1
            for i in range(1, self.n + 1):
                n_factorial *= i

            x_factorial = 1
            for i in range(1, x + 1):
                x_factorial *= i

            n_choose_k = 1
            for i in range(1, (self.n - x) + 1):
                n_choose_k *= i

            combination = n_factorial // (x_factorial * n_choose_k)
            cdf_value += combination * (self.p ** x) * (
                (1 - self.p) ** (self.n - x))

        return cdf_value
