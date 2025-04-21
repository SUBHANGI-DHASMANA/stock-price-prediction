import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from config import CHART_HEIGHT, CANDLESTICK_COLORS

def plot_raw_data(data):
    # Check if data is available
    if data is not None and not data.empty and len(data) > 0:
        # Create two tabs for different visualizations
        price_tab, volume_tab = st.tabs(["Price History", "Trading Volume"])

        with price_tab:
            st.subheader("Price History")
            plot_price_history(data)
            
            # Add candlestick chart as an alternative view
            st.subheader("Candlestick Chart")
            plot_candlestick(data)

        with volume_tab:
            st.subheader("Trading Volume")
            plot_volume(data)
            display_volume_statistics(data)
    else:
        st.error("No data available to display. Please check your stock selection.")

def plot_price_history(data):
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
        height=CHART_HEIGHT,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Add range selector
    add_range_selector(fig)

    st.plotly_chart(fig, use_container_width=True)

def plot_candlestick(data):
    candle_fig = go.Figure()

    # Add candlestick chart
    candle_fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC',
        increasing_line_color=CANDLESTICK_COLORS['increasing'],
        decreasing_line_color=CANDLESTICK_COLORS['decreasing']
    ))

    # Improve the layout
    candle_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=CHART_HEIGHT,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Add range selector
    add_range_selector(candle_fig)

    st.plotly_chart(candle_fig, use_container_width=True)

def plot_volume(data):
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
        height=CHART_HEIGHT,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Add range selector
    add_range_selector(volume_fig)

    st.plotly_chart(volume_fig, use_container_width=True)

def display_volume_statistics(data):
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

def add_range_selector(fig):
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

def plot_forecast(data, historical_predictions, future_predictions, n_years):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        name='Historical'
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_predictions['Date'],
        y=historical_predictions['Predicted_Close'],
        name='Historical Predictions'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_predictions['Date'],
        y=future_predictions['Predicted_Close'],
        name='Future Predictions'
    ))
    
    fig.update_layout(
        title=f'{n_years} Year Forecast', 
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_training_history(history):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=history.history['loss'], 
        name='Training Loss'
    ))
    
    fig.update_layout(
        title='Training History', 
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)