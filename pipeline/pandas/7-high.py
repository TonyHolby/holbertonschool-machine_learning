#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame and sorts it by the High price in
    descending order.
"""


def high(df):
    """
        Takes a pd.DataFrame and sorts it by the High price in descending
        order.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data.

        Returns:
            The sorted pd.DataFrame.
    """
    sorted_df = df.sort_values(by="High", ascending=False)

    return sorted_df
