import streamlit as st
from visualizations import plot_forecast, plot_training_history

def display_model_results(data, history, rmse, mape, historical_predictions, future_predictions, selected_stock, n_years):
    # Display metrics
    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"${rmse:.2f}")
    col2.metric("MAPE", f"{mape:.2f}%")

    # Plot training history (loss)
    plot_training_history(history)

    # Plot forecast
    plot_forecast(data, historical_predictions, future_predictions, n_years)

    # Download buttons
    provide_download_options(historical_predictions, future_predictions, selected_stock)

def provide_download_options(historical_predictions, future_predictions, selected_stock):
    col1, col2 = st.columns(2)
    
    for col, df, name in [(col1, future_predictions, 'future'),
                        (col2, historical_predictions, 'historical')]:
        col.download_button(
            label=f"Download {name.capitalize()} Predictions",
            data=df.to_csv(index=False),
            file_name=f"{selected_stock}_{name}_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )