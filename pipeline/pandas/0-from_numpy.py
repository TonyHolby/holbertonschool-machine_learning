#!/usr/bin/env python3
"""
    A function that creates a pd.DataFrame from a np.ndarray.
"""
import pandas as pd


def from_numpy(array):
    """
        Creates a pd.DataFrame from a np.ndarray.

        Args:
            array (np.ndarray): the np.ndarray from which the
                pd.DataFrame is created.

        Returns:
            The newly created pd.DataFrame.
    """
    number_of_cols = array.shape[1]
    columns = []
    for i in range(number_of_cols):
        columns.append(chr(ord('A') + i))

    df = pd.DataFrame(array, columns=columns)

    return df
