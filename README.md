# Stock Price Forecasting with LSTM

A Streamlit web application for forecasting stock prices using Long Short-Term Memory (LSTM) neural networks.

![Stock Forecasting App](https://raw.githubusercontent.com/username/stock-price-prediction/main/screenshot.png)

## Features

- **Real-time Stock Data**: Fetches the latest stock data from Yahoo Finance
- **Interactive Visualizations**: Dynamic charts with range selectors and interactive features
- **LSTM Model**: Deep learning model for time series forecasting
- **Performance Metrics**: RMSE and MAPE to evaluate prediction accuracy
- **Confidence Intervals**: Visualize prediction uncertainty
- **Training History**: Monitor model training performance

## How It Works

The application uses LSTM (Long Short-Term Memory) neural networks, a type of recurrent neural network well-suited for time series forecasting. The model:

1. **Preprocesses** historical stock data with normalization
2. **Trains** on sequences of 60 days to predict the next day's price
3. **Validates** on a separate dataset to prevent overfitting
4. **Forecasts** future stock prices for the selected time period
5. **Evaluates** performance using RMSE and MAPE metrics

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Select a stock from the dropdown menu

4. Choose the number of years for prediction (1-4)

5. View the forecasting results and performance metrics

## Model Architecture

The LSTM model architecture consists of:

- Input layer with 60 time steps (60 days of historical data)
- 3 LSTM layers with 100 units each
- Dropout layers (0.2) for regularization
- Dense output layer for prediction

## Performance Metrics

- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Measures the percentage difference between predicted and actual values

## Limitations

- Stock markets are influenced by many external factors that may not be captured in historical price data
- The model assumes patterns in historical data will continue in the future
- Extreme market events or black swan events cannot be predicted
- The model does not incorporate fundamental analysis or news sentiment

## Future Improvements

- Add more technical indicators as features (RSI, MACD, Bollinger Bands)
- Implement ensemble models for improved accuracy
- Add sentiment analysis from news and social media
- Include option for multivariate forecasting with related stocks or indices
- Implement backtesting framework for model evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [Yahoo Finance](https://finance.yahoo.com/) for the stock data
- [Plotly](https://plotly.com/) for interactive visualizations
