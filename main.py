import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Create session state for tracking previous selections
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

stocks = ('GOOG', 'AAPL', 'MSFT', 'JPM', "TSLA", "AMZN", "META", "NVDA")
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Check if stock or years changed
if st.session_state.prev_stock is not None and (st.session_state.prev_stock != selected_stock or st.session_state.prev_years != n_years):
    st.session_state.model_trained = False
    st.warning("Stock or prediction period changed. Please train the model again.")

# Update previous selections
st.session_state.prev_stock, st.session_state.prev_years = selected_stock, n_years

# LSTM is the default model
st.write("Using LSTM model for prediction")


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    # Check if data is available
    if data is not None and not data.empty and len(data) > 0:
        # Create two tabs for different visualizations
        price_tab, volume_tab = st.tabs(["Price History", "Trading Volume"])

        with price_tab:
            st.subheader("Price History")

            # Create a figure with subplots: main chart and volume
            fig = go.Figure()

            # Add price lines
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#2E86C1', width=2)
            ))

            # Add moving averages
            ma50 = data['Close'].rolling(window=50).mean()
            ma200 = data['Close'].rolling(window=200).mean()

            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=ma50,
                mode='lines',
                name='50-day MA',
                line=dict(color='#F39C12', width=1.5)
            ))

            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=ma200,
                mode='lines',
                name='200-day MA',
                line=dict(color='#E74C3C', width=1.5)
            ))

            # Improve the layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white',
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="YTD", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add candlestick chart as an alternative view
            st.subheader("Candlestick Chart")
            candle_fig = go.Figure()

            # Add candlestick chart
            candle_fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ))

            # Improve the layout
            candle_fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template='plotly_white',
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            # Add range selector
            candle_fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="YTD", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(candle_fig, use_container_width=True)

        with volume_tab:
            st.subheader("Trading Volume")

            # Calculate daily price change for coloring volume bars
            data['Daily_Change'] = data['Close'].diff()

            # Create a figure for volume
            volume_fig = go.Figure()

            # Add colored volume bars based on price movement
            colors = ['#EF5350' if val <= 0 else '#26A69A' for val in data['Daily_Change']]

            volume_fig.add_trace(go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='Volume',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(58, 71, 80, 0.2)', width=0.5)
                )
            ))

            # Add a moving average of volume
            vol_ma = data['Volume'].rolling(window=20).mean()

            volume_fig.add_trace(go.Scatter(
                x=data['Date'],
                y=vol_ma,
                name='20-day MA Volume',
                line=dict(color='#F39C12', width=2)
            ))

            # Improve the layout
            volume_fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Volume',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white',
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            # Add range selector
            volume_fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="YTD", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(volume_fig, use_container_width=True)

            # Add a table with summary statistics
            st.subheader("Volume Statistics")

            # Calculate volume statistics
            try:
                vol_stats = {
                    'Average Daily Volume': f"{int(data['Volume'].mean()):,}",
                    'Maximum Volume': f"{int(data['Volume'].max()):,}",
                    'Minimum Volume': f"{int(data['Volume'].min()):,}",
                    'Total Volume': f"{int(data['Volume'].sum()):,}"
                }
            except Exception as e:
                st.error(f"Error calculating volume statistics: {e}")
                vol_stats = {
                    'Average Daily Volume': 'N/A',
                    'Maximum Volume': 'N/A',
                    'Minimum Volume': 'N/A',
                    'Total Volume': 'N/A'
                }

            # Create a dataframe for display
            vol_stats_df = pd.DataFrame(list(vol_stats.items()), columns=['Metric', 'Value'])
            st.table(vol_stats_df)
    else:
        st.error("No data available to display. Please check your stock selection.")


plot_raw_data()

# Function for LSTM prediction
def predict_with_lstm(data, n_years):
    # Prepare data
    df = data.copy()
    df = df[['Date', 'Close']]
    df.set_index('Date', inplace=True)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create training dataset
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    # Create sequences
    sequence_length = 60
    x_train, y_train = [], []

    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model with progress bar
    epochs = 50
    progress_bar = st.progress(0)

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs, progress_bar):
            self.total_epochs = total_epochs
            self.progress_bar = progress_bar

        def on_epoch_end(self, epoch, logs=None):
            # Update progress bar
            # logs parameter is required by Keras but not used here
            self.progress_bar.progress((epoch + 1) / self.total_epochs)

    callback = ProgressCallback(epochs, progress_bar)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[callback])
    progress_bar.progress(1.0)

    # Prepare test data
    test_data = scaled_data[train_size - sequence_length:]
    x_test, y_test = [], []

    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i-sequence_length:i, 0])
        y_test.append(test_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    # Calculate MAPE
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    # Predict future values
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

    # Create DataFrames
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
    fig.update_layout(title='Training History', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical'))
    fig.add_trace(go.Scatter(
        x=st.session_state.historical_predictions['Date'],
        y=st.session_state.historical_predictions['Predicted_Close'],
        name='Historical Predictions'
    ))
    fig.add_trace(go.Scatter(
        x=st.session_state.future_predictions['Date'],
        y=st.session_state.future_predictions['Predicted_Close'],
        name='Future Predictions'
    ))
    fig.update_layout(title=f'{n_years} Year Forecast', height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    for col, df, name in [(col1, st.session_state.future_predictions, 'future'),
                         (col2, st.session_state.historical_predictions, 'historical')]:
        col.download_button(
            label=f"Download {name.capitalize()} Predictions",
            data=df.to_csv(index=False),
            file_name=f"{selected_stock}_{name}_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info('👆 Click "Train LSTM Model" to see predictions.')
