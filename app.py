import streamlit as st
from datetime import date

from config import STOCKS, START_DATE
from data_loader import load_data
from visualizations import plot_raw_data
from model import predict_with_lstm
import utils

# App title
st.title('Stock Forecast App')

# Initialize session state for tracking selections
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

# UI for stock selection and prediction years
selected_stock = st.selectbox('Select dataset for prediction', STOCKS)
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

# Load data
TODAY = date.today().strftime("%Y-%m-%d")
data_load_state = st.text('Loading data...')
data = load_data(selected_stock, START_DATE, TODAY)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot data
plot_raw_data(data)

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
    utils.display_model_results(
        data, 
        st.session_state.history,
        st.session_state.rmse,
        st.session_state.mape,
        st.session_state.historical_predictions,
        st.session_state.future_predictions,
        selected_stock,
        n_years
    )
else:
    st.info('👆 Click "Train LSTM Model" to see predictions.')