#!/usr/bin/env python3
"""
    A function that takes two pd.DataFrame objects, indexes both dataframes
    on their Timestamp columns, includes all timestamps from df2 (bitstamp)
    up to and including timestamp 1417411920, concatenates the selected rows
    from df2 to the top of df1 (coinbase) and dds keys to the concatenated
    data, labeling the rows from df2 as bitstamp and the rows from df1 as
    coinbase.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
        Takes two pd.DataFrame objects, indexes both dataframes on their
        Timestamp columns, includes all timestamps from df2 (bitstamp) up to
        and including timestamp 1417411920, concatenates the selected rows
        from df2 to the top of df1 (coinbase) and dds keys to the concatenated
        data, labeling the rows from df2 as bitstamp and the rows from df1 as
        coinbase.

        Args:
            df1 (pd.DataFrame): a pd.DataFrame containing the coinbase data.
            df2 (pd.DataFrame): a pd.DataFrame containing the bitstamp data.

        Returns:
            The concatenated pd.DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2_selected = df2.loc[:1417411920]
    concatenated_df = pd.concat([df2_selected, df1],
                                keys=["bitstamp", "coinbase"])

    return concatenated_df
