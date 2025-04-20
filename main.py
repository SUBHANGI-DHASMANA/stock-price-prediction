import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Constants
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
stocks = ('GOOG', 'AAPL', 'MSFT', 'JPM', "TSLA", "AMZN", "META", "NVDA")

# App title
st.title('Stock Forecast App')

# Initialize session state
if 'prev_stock' not in st.session_state:
    st.session_state.update({
        'prev_stock': None,
        'prev_years': None,
        'model_trained': False,
        'all_predictions': None,
        'historical_predictions': None,
        'future_predictions': None,
        'rmse': None,
        'mape': None,
        'history': None
    })

# User inputs
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Check if stock or years changed
if st.session_state.prev_stock is not None and (st.session_state.prev_stock != selected_stock or st.session_state.prev_years != n_years):
    st.session_state.model_trained = False
    st.warning("Stock or prediction period changed. Please train the model again.")

# Update previous selections
st.session_state.prev_stock, st.session_state.prev_years = selected_stock, n_years

# Data loading
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if data.empty:
        st.error(f"No data found for {ticker}. Try another stock or date range.")
        return pd.DataFrame()
    return data.reset_index()

data = load_data(selected_stock)

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    if data.empty:
        return st.error("No data available to display.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', line=dict(color='#2E86C1')))

    for window, color in [(50, '#F39C12'), (200, '#E74C3C')]:
        if len(data) >= window:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'].rolling(window).mean(),
                name=f'{window}-day MA',
                line=dict(color=color, width=1)
            ))

    fig.update_layout(
        title=f'{selected_stock} Historical Price Data',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# LSTM Model
def predict_with_lstm(data, n_years):
    if data.empty:
        # Create empty DataFrames with proper structure
        empty_df1 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        empty_df2 = pd.DataFrame(columns=['Date', 'Predicted_Close', 'Actual_Close'])
        empty_df3 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        # Create a dummy history object
        dummy_history = type('obj', (object,), {'history': {'loss': [], 'val_loss': []}})
        # Return the properly structured objects
        return empty_df1, empty_df2, empty_df3, 0, 0, dummy_history

    # Keep a copy of the original data with dates
    data_with_dates = data.copy()
    
    # Create a dataframe with Close prices only
    df = data[['Close']].copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    sequence_length = 60
    if len(scaled_data) <= sequence_length:
        st.error(f"Need more than {sequence_length} data points.")
        # Create empty DataFrames with proper structure
        empty_df1 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        empty_df2 = pd.DataFrame(columns=['Date', 'Predicted_Close', 'Actual_Close'])
        empty_df3 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        # Create a dummy history object
        dummy_history = type('obj', (object,), {'history': {'loss': [], 'val_loss': []}})
        # Return the properly structured objects
        return empty_df1, empty_df2, empty_df3, 0, 0, dummy_history

    # Prepare data
    train_size = int(len(scaled_data) * 0.8)
    train_data, val_data = scaled_data[:train_size], scaled_data[train_size - sequence_length:]

    def create_sequences(data):
        x, y = [], []
        for i in range(sequence_length, len(data)):
            x.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(train_data)
    x_val, y_val = create_sequences(val_data)

    if len(x_train) == 0:
        # Create empty DataFrames with proper structure
        empty_df1 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        empty_df2 = pd.DataFrame(columns=['Date', 'Predicted_Close', 'Actual_Close'])
        empty_df3 = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        # Create a dummy history object
        dummy_history = type('obj', (object,), {'history': {'loss': [], 'val_loss': []}})
        # Return the properly structured objects
        return empty_df1, empty_df2, empty_df3, 0, 0, dummy_history

    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1)) if len(x_val) > 0 else x_train[-int(len(x_train)*0.1):]
    y_val = y_val if len(y_val) > 0 else y_train[-int(len(y_train)*0.1):]

    # Build model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    epochs = 50
    progress_bar = st.progress(0)

    # Progress callback for training
    class ProgressCallback(tensorflow.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # logs parameter is required by Keras but not used here
            progress_bar.progress((epoch + 1) / epochs)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[ProgressCallback()],
        verbose=0
    )
    progress_bar.progress(1.0)

    # Make predictions
    try:
        # Transform input data
        inputs = scaled_data.copy()

        # Create test sequences
        x_test = []
        for i in range(sequence_length, len(inputs)):
            x_test.append(inputs[i-sequence_length:i, 0])

        # Convert to numpy array and reshape
        x_test = np.array(x_test)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Make predictions
        raw_predictions = model.predict(x_test)

        # Inverse transform to get actual prices
        predicted_prices = np.zeros((len(raw_predictions), 1))
        predicted_prices[:, 0] = raw_predictions.flatten()
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Get actual prices for comparison
        actual_prices = df['Close'].values[sequence_length:sequence_length+len(predicted_prices)]

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        # Create empty arrays
        predicted_prices = np.array([]).reshape(-1, 1)
        actual_prices = np.array([])

    # Calculate metrics
    try:
        # Check if we have enough data for metrics
        if len(actual_prices) > 0 and len(predicted_prices) > 0:
            # Ensure dimensions match
            pred_flat = predicted_prices.flatten()
            if len(pred_flat) > len(actual_prices):
                pred_flat = pred_flat[:len(actual_prices)]
            elif len(actual_prices) > len(pred_flat):
                actual_prices = actual_prices[:len(pred_flat)]

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(actual_prices, pred_flat))

            # Calculate MAPE with handling for division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.nanmean(np.abs((actual_prices - pred_flat) / actual_prices)) * 100
                # Replace NaN or inf values
                if np.isnan(mape) or np.isinf(mape):
                    mape = 0.0
        else:
            rmse, mape = 0.0, 0.0
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        rmse, mape = 0.0, 0.0

    # Future predictions
    try:
        # Get the last sequence for future predictions
        last_sequence = scaled_data[-sequence_length:].copy()
        future_preds = []

        # Reshape for model input
        current_batch = last_sequence.reshape(1, sequence_length, 1)

        # Generate future predictions
        for _ in range(n_years * 365):
            # Predict next value
            pred_result = model.predict(current_batch, verbose=0)
            
            # Extract the prediction value
            pred = pred_result[0][0]
            
            future_preds.append(pred)

            # Update the sequence with the new prediction
            current_batch = np.append(current_batch[:,1:,:], [[[pred]]], axis=1)

        # Convert predictions to actual prices
        future_preds_array = np.array(future_preds).reshape(-1, 1)
        future_preds = scaler.inverse_transform(future_preds_array)

        # Generate future dates
        last_date = data_with_dates['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_years * 365)

    except Exception as e:
        st.error(f"Error generating future predictions: {e}")
        # Create empty arrays and dates
        future_preds = np.array([]).reshape(-1, 1)
        future_dates = pd.date_range(start=data_with_dates['Date'].iloc[-1] + pd.Timedelta(days=1), periods=1)

    # Create DataFrames
    try:
        if len(predicted_prices) > 0:
            # Get dates for historical predictions
            historical_dates = data_with_dates['Date'].iloc[sequence_length:sequence_length+len(predicted_prices)]
            
            # Create historical predictions DataFrame
            historical_df = pd.DataFrame({
                'Date': historical_dates,
                'Predicted_Close': predicted_prices.flatten(),
                'Actual_Close': actual_prices
            })
        else:
            # Create an empty DataFrame with the correct structure
            historical_df = pd.DataFrame(columns=['Date', 'Predicted_Close', 'Actual_Close'])
    except Exception as e:
        st.error(f"Error creating historical DataFrame: {e}")
        # Create an empty DataFrame with the correct structure
        historical_df = pd.DataFrame(columns=['Date', 'Predicted_Close', 'Actual_Close'])

    # Create future predictions DataFrame
    try:
        if len(future_preds) > 0:
            # Create future predictions DataFrame
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Close': future_preds.flatten()
            })
        else:
            # Create an empty DataFrame with the correct structure
            future_df = pd.DataFrame(columns=['Date', 'Predicted_Close'])
    except Exception as e:
        st.error(f"Error creating future predictions DataFrame: {e}")
        # Create an empty DataFrame with the correct structure
        future_df = pd.DataFrame(columns=['Date', 'Predicted_Close'])

    # Combine historical and future predictions
    all_predictions = pd.concat([historical_df, future_df], ignore_index=True)
    
    return all_predictions, historical_df, future_df, rmse, mape, history

# Model training UI
st.subheader('LSTM Forecast')

col1, col2 = st.columns([3, 1])
with col1:
    train_button = st.button('Train LSTM Model', use_container_width=True)
with col2:
    if st.button('Reset', use_container_width=True):
        st.session_state.model_trained = False
        st.experimental_rerun()

if train_button or st.session_state.model_trained:
    if not st.session_state.model_trained:
        with st.spinner('Training LSTM model...'):
            results = predict_with_lstm(data, n_years)
            st.session_state.update({
                'model_trained': True,
                'all_predictions': results[0],
                'historical_predictions': results[1],
                'future_predictions': results[2],
                'rmse': results[3],
                'mape': results[4],
                'history': results[5]
            })
        st.success('LSTM model training complete!')

# Display results
if st.session_state.model_trained:
    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"${st.session_state.rmse:.2f}")
    col2.metric("MAPE", f"{st.session_state.mape:.2f}%")

    # Plot training history
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.history.history['loss'], name='Training Loss'))
    if 'val_loss' in st.session_state.history.history:
        fig.add_trace(go.Scatter(y=st.session_state.history.history['val_loss'], name='Validation Loss'))
    fig.update_layout(title='Training History', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast plot
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        name='Historical Data',
        line=dict(color='#2E86C1')
    ))
    
    # Plot historical predictions
    if not st.session_state.historical_predictions.empty:
        fig.add_trace(go.Scatter(
            x=st.session_state.historical_predictions['Date'],
            y=st.session_state.historical_predictions['Predicted_Close'],
            name='Historical Predictions',
            line=dict(color='#F39C12')
        ))
    
    # Plot future predictions
    if not st.session_state.future_predictions.empty:
        fig.add_trace(go.Scatter(
            x=st.session_state.future_predictions['Date'],
            y=st.session_state.future_predictions['Predicted_Close'],
            name='Future Predictions',
            line=dict(color='#E74C3C')
        ))
    
    fig.update_layout(
        title=f'{selected_stock} {n_years} Year Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    for col, df, name in [(col1, st.session_state.future_predictions, 'future'),
                         (col2, st.session_state.historical_predictions, 'historical')]:
        if not df.empty:
            col.download_button(
                label=f"Download {name.capitalize()} Predictions",
                data=df.to_csv(index=False),
                file_name=f"{selected_stock}_{name}_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            col.warning(f"No {name} predictions available for download")
else:
    st.info('👆 Click "Train LSTM Model" to see predictions.')