#!/usr/bin/env python3
"""
    A script that trains an LSTM model for Bitcoin price forecasting:
    - Loads preprocessed hourly Bitcoin data from a Numpy file.
    - Creates supervised datasets (features and target) using a sliding
      window of past observations (lookback).
    - Splits the dataset into training, validation, and test sets.
    - Defines and trains an LSTM neural network to predict the next hour
      Close price using both the Close and Weighted Price as input features.
    - Uses callbacks for early stopping and learning rate scheduling.
    - Saves the trained model in Keras format.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


LOOKBACK = 24
FEATURES = 2
BATCH_SIZE = 64
EPOCHS = 20


def create_dataset(data, lookback=24):
    """
        Transforms the time series data into a supervised learning dataset.

        Args:
            data (np.ndarray): A Numpy array containing the selected features.
            lookback (int): the number of past time steps used as input.

        Returns:
            X: the input sequences of shape (samples, lookback, features).
            y: the target values (next hour Close price).
    """
    X, y = [], []
    close_idx = 0
    weighted_idx = 1

    for i in range(lookback, len(data)):
        features = data[i-lookback:i, [close_idx, weighted_idx]]
        X.append(features)
        y.append(data[i, close_idx])

    return np.array(X), np.array(y)


def split_datasets(X, y, train_ratio=0.7, val_ratio=0.2):
    """
        Splits dataset into training, validation, and test sets.

        Args:
            X (np.ndarray): the input features.
            y (np.ndarray): the target values.
            train_ratio (float): the proportion of samples for training set.
            val_ratio (float): the proportion of samples for validation set.

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test): tuples of
                Numpy arrays.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (X[:train_end], y[:train_end]), \
           (X[train_end:val_end], y[train_end:val_end]), \
           (X[val_end:], y[val_end:])


def main():
    """
        Runs the training of the LSTM model:
        - Loads preprocessed hourly Bitcoin data.
        - Creates input and output datasets.
        - Splits data into training, validation, and test sets.
        - Defines, compiles, and trains an LSTM network.
        - Saves the trained model.
    """
    data = np.load("btc_hourly.npy")
    X, y = create_dataset(data, LOOKBACK)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_datasets(X, y)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model = Sequential([
        LSTM(96, input_shape=(LOOKBACK, FEATURES)),
        Dense(48, activation="relu"),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="mse")
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=3,
                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  patience=2,
                                  min_lr=1e-7)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=[early_stop, reduce_lr])
    model.save("btc_forecast_model.keras")


if __name__ == "__main__":
    main()
