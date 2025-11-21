#!/usr/bin/env python3
"""
    A function that Takes two pd.DataFrame objects, rearranges the MultiIndex
    so that Timestamp is the first level, concatenates the bitstamp and
    coinbase tables from timestamps 1417411980 to 1417417980, inclusive and
    adds keys to the data, labeling rows from df2 as bitstamp and rows from
    df1 as coinbase.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
        Takes two pd.DataFrame objects, rearranges the MultiIndex so that
        Timestamp is the first level, concatenates the bitstamp and coinbase
        tables from timestamps 1417411980 to 1417417980, inclusive and adds
        keys to the data, labeling rows from df2 as bitstamp and rows from
        df1 as coinbase.

        Args:
            df1 (pd.DataFrame): a pd.DataFrame containing the coinbase data.
            df2 (pd.DataFrame): a pd.DataFrame containing the bitstamp data.

        Returns:
            The concatenated pd.DataFrame in chronological order
    """
    df1 = index(df1)
    df2 = index(df2)
    df1_selected = df1.loc[1417411980:1417417980]
    df2_selected = df2.loc[1417411980:1417417980]
    concatenated_df = pd.concat([df2_selected, df1_selected],
                                keys=["bitstamp", "coinbase"])
    concatenated_df = concatenated_df.reorder_levels([1, 0])
    concatenated_df = concatenated_df.sort_index()

    return concatenated_df
