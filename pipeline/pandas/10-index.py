#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame and sets the Timestamp column as
    the index of the dataframe.
"""


def index(df):
    """
        Takes a pd.DataFrame and sets the Timestamp column as the index of
        the dataframe.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data and column
                labels.

        Returns:
            The modified pd.DataFrame.
    """
    indexed_df = df.set_index("Timestamp")

    return indexed_df
