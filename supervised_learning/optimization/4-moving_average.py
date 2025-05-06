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
    weighted_ema = 0
    ema_list = []

    for t, element in enumerate(data, 1):
        weighted_ema = beta * weighted_ema + (1 - beta) * element
        corrected_ema = weighted_ema / (1 - beta ** t)
        ema_list.append(corrected_ema)

    return ema_list
