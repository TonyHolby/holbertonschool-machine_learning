#!/usr/bin/env python3
"""
    A function that calculates the weighted moving average of a data set.
"""


def moving_average(data, beta):
    """
        Calculates the weighted moving average of a data set.

        Args:
            data (list): the list of data to calculate the moving average of.
            beta (float): the weight used for the moving average.

        Returns:
            A list containing the moving averages of data.
    """
    weighted_ema = data[0]
    ema_list = []

    for element in data:
        weighted_ema = beta * weighted_ema + (1-beta) * element
        ema_list.append(weighted_ema)

    return ema_list
