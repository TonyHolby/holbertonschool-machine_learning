#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame, extracts the columns High, Low,
    Close and Volume_(BTC) and selects every 60th row from these columns.
"""


def slice(df):
    """
        Takes a pd.DataFrame, extracts the columns High, Low, Close and
        Volume_(BTC) and selects every 60th row from these columns.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the columns.

        Returns:
            The sliced pd.DataFrame.
    """
    selected = df[["High", "Low", "Close", "Volume_(BTC)"]]
    sliced = selected.iloc[::60]

    return sliced
