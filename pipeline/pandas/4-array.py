#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame as input, selects the last 10 rows
    of the High and Close columns and converts these selected values into a
    numpy.ndarray.
"""


def array(df):
    """
        Takes a pd.DataFrame as input, selects the last 10 rows of the High
        and Close columns and converts these selected values into a
        numpy.ndarray.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing columns named High
                and Close.

        Returns:
            The numpy.ndarray.
    """
    last_values = df[["High", "Close"]].tail(10)
    numpy_array = last_values.to_numpy()

    return numpy_array
