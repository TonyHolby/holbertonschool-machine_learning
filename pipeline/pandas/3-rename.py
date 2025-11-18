#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame as input, rename the Timestamp column
    to Datetime, convert the timestamp values to datatime values and display
    only the Datetime and Close column.
"""
import pandas as pd


def rename(df):
    """
        Takes a pd.DataFrame as input, rename the Timestamp column to Datetime,
        convert the timestamp values to datatime values and display only the
        Datetime and Close column.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing a column named
                Timestamp.

        Returns:
            The modified pd.DataFrame.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    df = df[["Datetime", "Close"]]

    return df
