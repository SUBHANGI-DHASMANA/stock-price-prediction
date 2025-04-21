# Configuration settings for the Stock Forecast App

# Start date for historical data
START_DATE = "2015-01-01"

# Available stocks to select from
STOCKS = ('GOOG', 'AAPL', 'MSFT', 'JPM', "TSLA", "AMZN", "META", "NVDA")

# LSTM model parameters
SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 32

# Visualization settings
CHART_HEIGHT = 500
CANDLESTICK_COLORS = {
    'increasing': '#26A69A',
    'decreasing': '#EF5350'
}