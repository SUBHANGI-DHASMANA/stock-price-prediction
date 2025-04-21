import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st

from data_loader import prepare_data_for_model
from config import SEQUENCE_LENGTH, TRAIN_TEST_SPLIT, LSTM_UNITS, DROPOUT_RATE, EPOCHS, BATCH_SIZE

def predict_with_lstm(data, n_years):
    # Prepare data
    df = prepare_data_for_model(data)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create training dataset
    train_size = int(len(scaled_data) * TRAIN_TEST_SPLIT)
    train_data = scaled_data[:train_size]

    # Create sequences
    x_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and train model
    model, history = build_and_train_model(x_train, y_train)

    # Prepare test data
    test_data = scaled_data[train_size - SEQUENCE_LENGTH:]
    x_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions on test data
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate metrics
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    # Predict future values
    future_predictions = predict_future(model, scaled_data, scaler, n_years, SEQUENCE_LENGTH)

    # Create DataFrames for visualization
    last_date = df.index[-1]
    prediction_dates = pd.date_range(start=last_date, periods=len(predictions))
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_years * 365)

    historical_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Close': predictions.flatten(),
        'Actual_Close': y_test_actual.flatten()
    })

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_predictions.flatten()
    })

    all_predictions = pd.concat([historical_df, future_df])

    return all_predictions, historical_df, future_df, rmse, mape, history

def create_sequences(data, sequence_length):
    x, y = [], []
    
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
        
    return np.array(x), np.array(y)

def build_and_train_model(x_train, y_train):
    # Build model
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(units=LSTM_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model with progress bar
    progress_bar = st.progress(0)

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs, progress_bar):
            self.total_epochs = total_epochs
            self.progress_bar = progress_bar

        def on_epoch_end(self, epoch, logs=None):
            # Update progress bar
            self.progress_bar.progress((epoch + 1) / self.total_epochs)

    callback = ProgressCallback(EPOCHS, progress_bar)

    history = model.fit(
        x_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=0, 
        callbacks=[callback]
    )
    
    progress_bar.progress(1.0)
    
    return model, history

def predict_future(model, scaled_data, scaler, n_years, sequence_length):
    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []

    current_batch = last_sequence.reshape((1, sequence_length, 1))

    for i in range(n_years * 365):
        current_pred = model.predict(current_batch)[0]
        future_predictions.append(current_pred[0])

        # Update sequence
        current_batch = np.append(current_batch[:, 1:, :],
                                [[current_pred]],
                                axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions