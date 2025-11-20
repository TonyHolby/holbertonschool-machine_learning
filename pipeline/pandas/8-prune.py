#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame and removes any entries where Close
    has NaN values.
"""


def prune(df):
    """
        Takes a pd.DataFrame and removes any entries where Close has NaN
        values.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data and column
                labels.

        Returns:
            The modified pd.DataFrame.
    """
    pruned_df = df.dropna(subset=["Close"])

    return pruned_df
