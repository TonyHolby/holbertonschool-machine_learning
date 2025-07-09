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

    def z_score(self, x):
        """
            Calculates the z-score of a given x-value.

            Args:
                x (float): the x-value.

            Returns:
                The z-score of x.
        """
        z_score = (x - self.mean) / self.stddev

        return z_score

    def x_value(self, z):
        """
            Calculates the x-value of a given z-score.

            Args:
                z (float): the z-score.

            Returns:
                The x-value of z.
        """
        x_value = (z * self.stddev) + self.mean

        return x_value

    def pdf(self, x):
        """
            Calculates the value of the PDF for a given x-value.

            Args:
                x (float): the x-value.

            Returns:
                The PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536
        pdf_value = (1 / (self.stddev * ((2 * pi) ** 0.5))) * (
            e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

        return pdf_value

    def cdf(self, x):
        """
            Calculates the value of the CDF for a given x-value.

            Args:
                x (float): the x-value.

            Returns:
                The CDF value for x.
        """
        pi = 3.1415926536
        z = self.z_score(x) / (2 ** 0.5)
        erf = (2 / (pi ** 0.5)) * (z - ((z ** 3) / 3) + (
            (z ** 5) / 10) - ((z ** 7) / 42) + ((z ** 9) / 216))
        cdf_value = 0.5 * (1 + erf)

        return cdf_value
