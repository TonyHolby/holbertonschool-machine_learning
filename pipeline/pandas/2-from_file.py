#!/usr/bin/env python3
"""
    A function that loads data from a file as a pd.DataFrame.
"""
import pandas as pd


def from_file(filename, delimiter):
    """
        Loads data from a file as a pd.DataFrame.

        Args:
            filename (str): the file to load from.
            delimiter (str): the column separator.

        Returns:
            The loaded pd.DataFrame.
    """
    df = pd.read_csv(filename, delimiter=delimiter)

    return df
