import pandas as pd
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import logging

logger = logging.getLogger(__name__)

def create_sequences(data, input_steps, output_steps):
    """
    Creates sequences X and Y for RNN models.
    data: array of values (one-dimensional)
    input_steps: input window
    output_steps: prediction horizon
    """
    x, y = [], []
    # Scroll until (len(data) - input_steps - output_steps + 1)
    for i in range(len(data) - input_steps - output_steps + 1):
        # Input sequence
        x.append(data[i : i + input_steps])
        # Output sequence (multi-step prediction)
        y.append(data[i + input_steps : i + input_steps + output_steps])
    return np.array(x), np.array(y)

# ---------------------------------------------------------------------
# ----------------------- Neural networks (GRU, LSTM) -----------------
# ---------------------------------------------------------------------

def train_gru(data, input_steps, output_steps, epochs=20, batch_size=32):
    """
    Trains a GRU model on a single time series.
    """
    values = data["value"].values  # Extract values
    x, y = create_sequences(values, input_steps, output_steps)  # Create sequences
    # Configure the GRU model
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, activation='tanh', input_shape=(input_steps, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_steps)  # Dense con output_steps
    ])
    model.compile(optimizer='adam', loss='mse')

    # Add an axis to represent the feature dimension
    x_expanded = np.expand_dims(x, axis=-1)  # shape: (n_batch, input_steps, 1)

    # Model training
    model.fit(x_expanded, y, epochs=epochs, batch_size=batch_size, verbose=0)
    logger.info("GRU training completed.")

    # Prediction on the last training window (optional)
    predictions = model.predict(x_expanded[-1:], verbose=0)[0]  # shape: (output_steps,)
    timestamps = pd.date_range(
        start=data['timestamp'].iloc[-1], periods=output_steps + 1, freq='T'
    )[1:]

    return model, predictions, timestamps

def train_lstm(data, input_steps, output_steps, epochs=20, batch_size=32):
    """
    Trains an LSTM model on a single time series.
    """
    values = data["value"].values
    x, y = create_sequences(values, input_steps, output_steps)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='tanh', input_shape=(input_steps, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_steps)  # Dense con output_steps
    ])
    model.compile(optimizer='adam', loss='mse')

    x_expanded = np.expand_dims(x, axis=-1)  # (n_batch, input_steps, 1)
    model.fit(x_expanded, y, epochs=epochs, batch_size=batch_size, verbose=0)
    logger.info("LSTM training completed.")

    predictions = model.predict(x_expanded[-1:], verbose=0)[0]  # shape: (output_steps,)
    timestamps = pd.date_range(
        start=data['timestamp'].iloc[-1], periods=output_steps + 1, freq='T'
    )[1:]

    return model, predictions, timestamps

# ---------------------------------------------------------------------
# ----------------------- Traditional models (ARIMA, PROPHET) -------
# ---------------------------------------------------------------------

def train_arima(data, output_steps):
    """
    Esegue ARIMA su una singola serie temporale.
    """
    values = data['value'].values

    # Modello ARIMA
    model = ARIMA(values, order=(5, 1, 0))
    model_fit = model.fit()

    # Previsioni future
    predictions = model_fit.forecast(steps=output_steps)
    timestamps = pd.date_range(
        start=data['timestamp'].iloc[-1], periods=output_steps + 1, freq='T'
    )[1:]

    logger.info("ARIMA training completed.")
    return model_fit, predictions, timestamps

def train_prophet(data, output_steps):
    """
    Trains a Prophet model on a single time series.
    """
    df = data.rename(columns={"timestamp": "ds", "value": "y"})  # rename columns
    model = Prophet()
    model.fit(df)

    # Create a dataset for future predictions
    future = model.make_future_dataframe(periods=output_steps, freq='T')
    forecast = model.predict(future)

    # Predictions and timestamps
    predictions = forecast['yhat'].iloc[-output_steps:].values
    timestamps = future['ds'].iloc[-output_steps:].values

    logger.info("Prophet training completed.")
    return model, predictions, timestamps
