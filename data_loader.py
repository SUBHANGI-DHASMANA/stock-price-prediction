import streamlit as st
import pandas as pd
import yfinance as yf

@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    
    # Calculate daily price change for coloring volume bars
    if not data.empty and len(data) > 0:
        data['Daily_Change'] = data['Close'].diff()
    
    return data

def prepare_data_for_model(data):
    df = data.copy()
    df = df[['Date', 'Close']]
    df.set_index('Date', inplace=True)
    return df