import streamlit as st
import pandas as pd
import os
import sys
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Robust Path Setup ---
def setup_path():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
    except NameError:
        project_root = os.getcwd()
        if 'dashboard' in project_root:
            project_root = project_root.split('dashboard')[0]
    return project_root

# --- Data Loading ---
@st.cache_data
def load_data(root_path):
    """Loads and caches the final feature data from the output CSV."""
    csv_path = os.path.join(root_path, 'output', 'final_linguistic_features.csv')
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df['best_score'] = (df['sentiment_score_mda'] - df['complexity_delta'] - df['risk_text_change_score_risk']).fillna(0)
        return df
    except FileNotFoundError:
        return None

# --- Prediction Model ---
def predict_stock_price(ticker):
    """Fetches historical stock data and returns a 6-month forecast plot."""
    # Fetch data
    data = yf.download(ticker, start="2020-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    # yfinance can return a multi-level column index. We flatten it here.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data.reset_index(inplace=True)
    
    # Prepare for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    # Create and fit model
    m = Prophet()
    m.fit(df_train)
    
    # Make future dataframe
    future = m.make_future_dataframe(periods=180)
    forecast = m.predict(future)
    
    # Plot forecast
    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title=f'{ticker} 6-Month Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)'
    )
    return fig
