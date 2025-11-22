#!/usr/bin/env python3
"""
    A script that visualizes the pd.DataFrame, removes the column
    Weighted_Price, renames the column Timestamp to Date, convert
    the timestamp values to date values, indexes the data frame on
    Date, sets the missing values in Close to the previous row value,
    sets the missing values in High, Low, and Open to the same row's
    Close value, sets the missing values in Volume_(BTC) and
    Volume_(Currency) to 0, and plots the data from 2017.
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
if "Weighted_Price" in df.columns:
    df = df.drop(columns=["Weighted_Price"])

df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")
df["Close"] = df["Close"].fillna(method="ffill")

for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])

for col in ["Volume_(BTC)", "Volume_(Currency)"]:
    df[col] = df[col].fillna(0)

df_2017 = df.loc["2017":]
daily = df_2017.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

print(daily)

daily.plot(figsize=(8, 6))
plt.tight_layout()
plt.show()
