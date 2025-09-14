# Time Series Forecasting

## Description of the project

This project allows to use a Long Short-Term Memory (LSTM) to predict Bitcoin prices based on historical hourly data. The model is trained on Coinbase and Bitstamp datasets and uses the previous 24 hours to predict the closing price of the next hour.

## Project structure:

time_series/
│
├─ preprocess_data.py         # Script to clean and prepare the data
├─ forecast_btc.py            # Script to create, train, and validate the RNN model
├─ coinbase.csv               # Raw Coinbase dataset
├─ bitstamp.csv               # Raw Bitstamp dataset
├─ btc_hourly.npy             # Preprocessed data saved
├─ btc_forecast_model.keras   # Trained Keras model
└─ README.md                  # README file

## Requirements:

The files were interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9).
The files were executed with numpy (version 1.25.2), tensorflow (version 2.15) and pandas (version 2.2.2).

## Steps:

## 1. Data Preprocessing:

The preprocess_data.py script:
Loads and merges the CSV files (coinbase.csv and bitstamp.csv).
Aggregates data to hourly intervals.
Normalizes the data using a MinMaxScaler.
Selects relevant features: Close and Weighted prices.
Saves the preprocessed data to btc_hourly.npy.

### Run the script:

python preprocess_data.py

### Outputs:

btc_hourly.npy : preprocessed hourly data ready for training.

## 2. Model Training and Validation:

The forecast_btc.py script:
Loads the preprocessed data.
Creates a dataset using a sliding window of 24 hours to predict the next hour’s closing price.
Splits the data into train and validation sets.
Builds a Keras LSTM model.
Trains the model using Mean Squared Error (MSE) as the loss function.
Saves the trained model to btc_forecast_model.keras.

### Run the script:

python forecast_btc.py

### Outputs:

btc_forecast_model.keras : trained LSTM model.

### Visualization of training and validation MSE curves

![Training Metrics](model_training.png)

# Author

Tony NEMOUTHE
