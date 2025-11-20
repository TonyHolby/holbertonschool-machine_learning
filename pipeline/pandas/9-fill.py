#!/usr/bin/env python3
"""
    A function that takes a pd.DataFrame and removes the Weighted_Price
    column, fills missing values in the Close column with the previous
    row's value, fills missing values in the High, Low, and Open columns
    with the corresponding Close value in the same row and sets missing
    values in Volume_(BTC) and Volume_(Currency) to 0.
"""


def fill(df):
    """
        Takes a pd.DataFrame and removes the Weighted_Price column, fills
        missing values in the Close column with the previous row's value,
        fills missing values in the High, Low, and Open columns with the
        corresponding Close value in the same row and sets missing values
        in Volume_(BTC) and Volume_(Currency) to 0.

        Args:
            df (pd.DataFrame): a pd.DataFrame containing the data and column
                labels.

        Returns:
            The modified pd.DataFrame.
    """
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    for col in ["High", "Low", "Open"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
