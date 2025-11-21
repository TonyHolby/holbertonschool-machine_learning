#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame and computes descriptive
    statistics for all columns except the Timestamp column.
"""


def analyze(df):
    """
        Takes a pd.DataFrame and computes descriptive statistics
        for all columns except the Timestamp column.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data
                and column labels.

        Returns:
            A new pd.DataFrame containing these statistics.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    statistics_info = df.describe()

    return statistics_info
