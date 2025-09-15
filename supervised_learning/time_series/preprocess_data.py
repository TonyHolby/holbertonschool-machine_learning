#!/usr/bin/env python3
"""
    A script that preprocess the raw data from Coinbase and Bitstamp datasets:
    - Loads and merges the datasets.
    - Cleans missing values and remove duplicates.
    - Converts UNIX timestamps to datetime format.
    - Aggregates data from minute-level to hourly frequency.
    - Normalizes selected features using Min-Max scaling.
    - Saves hourly timestamps and normalized features as Numpy arrays.
"""
import pandas as pd
import numpy as np


def load_and_merge(datasets):
    """
        Loads and merges raw data from Coinbase and Bitstamp datasets.

        Args:
            datasets (list): a list containing CSV file paths of Coinbase
                             and Bitstamp datasets.

        Returns:
            merged_data: the preprocessed data.
    """
    dataframes = []
    for file in datasets:
        df = pd.read_csv(
            file,
            header=0,
            names=["timestamp",
                   "open",
                   "high",
                   "low",
                   "close",
                   "volume_btc",
                   "volume_currency",
                   "weighted_price"],
            dtype={"timestamp": "int64",
                   "open": "float64",
                   "high": "float64",
                   "low": "float64",
                   "close": "float64",
                   "volume_btc": "float64",
                   "volume_currency": "float64",
                   "weighted_price": "float64"},
            skiprows=1,
            low_memory=False
        )

        df = df.dropna()
        dataframes.append(df)

    merged_data = pd.concat(dataframes).\
        sort_values("timestamp").reset_index(drop=True)
    merged_data = merged_data.drop_duplicates()

    return merged_data


def aggregate_to_hourly(df):
    """
        Aggregates minute-level data to hourly frequency.

        Args:
            df (pandas.DataFrame): the cleaned dataframe at the minute level.

        Returns:
            hourly: the pandas DataFrame resampled to hourly intervals,
                with the following aggregation rules:
                    open: first value of the hour
                    high: maximum within the hour
                    low: minimum within the hour
                    close: last value of the hour
                    volume_btc: sum over the hour
                    volume_currency: sum over the hour
                    weighted_price: mean over the hour
    """
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    df = df[df.index >= "2017-01-01"]

    hourly = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_btc": "sum",
        "volume_currency": "sum",
        "weighted_price": "mean"
    }).dropna()

    return hourly


def normalize_features(df, feature_cols):
    """
        Normalizes selected features using Min-Max scaling.

        Args:
            df (pandas.DataFrame): the hourly aggregated dataframe.
            feature_cols (list): the columns to normalize.

        Returns:
            The normalized feature array with values scaled between 0 and 1.
    """
    features = df[feature_cols].values.astype(np.float64)
    min_values = features.min(axis=0)
    max_values = features.max(axis=0)
    scaled_data = (features - min_values) / (max_values - min_values)

    return scaled_data


def main():
    """
        Runs the data preprocessing:
            - Loads and merges Coinbase and Bitstamp datasets.
            - Aggregates data to hourly frequency.
            - Normalizes selected features (Close, Weighted_Price).
            - Saves the normalized features as a Numpy array.
    """
    datasets = [
        "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",
        "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    ]

    data = load_and_merge(datasets)
    hourly_data = aggregate_to_hourly(data)

    feature_cols = ["close", "weighted_price"]
    features_scaled = normalize_features(hourly_data, feature_cols)

    np.save("btc_hourly.npy", features_scaled)


if __name__ == "__main__":
    main()
