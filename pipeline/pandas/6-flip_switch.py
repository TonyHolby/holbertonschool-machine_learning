#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame, sorts the data in reverse
    chronological order and transposes the sorted dataframe.
"""


def flip_switch(df):
    """
        Takes a pd.DataFrame, sorts the data in reverse chronological
        order and transposes the sorted dataframe.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data.

        Returns:
            The transformed pd.DataFrame.
    """
    sorted_df = df.sort_index(ascending=False)
    transformed_df = sorted_df.T

    return transformed_df
