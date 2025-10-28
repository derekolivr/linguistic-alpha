import pandas as pd
import yfinance as yf
import numpy as np
import os
import sys

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from analysis.helpers import find_project_root

def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def calculate_volatility(stock_data, window=90):
    """Calculate rolling volatility for each stock."""
    # Use log returns for volatility calculation
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
    return volatility.rename('volatility')

if __name__ == "__main__":
    # 1. Load pre-calculated linguistic features from the pipeline output
    project_root = find_project_root()
    output_dir = os.path.join(project_root, 'output')
    
    try:
        core_features_df = pd.read_csv(os.path.join(output_dir, 'core_features.csv'))
        mda_features_df = pd.read_csv(os.path.join(output_dir, 'mda_features.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the main pipeline script ('run_pipeline.py') first to generate feature files.")
        exit()

    # Combine features for backtesting
    features_df = pd.concat([core_features_df, mda_features_df], ignore_index=True)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # 2. Get stock data
    tickers = features_df['ticker'].unique().tolist()
    start_date = features_df['date'].min() - pd.Timedelta(days=1)
    end_date = features_df['date'].max() + pd.Timedelta(days=91)

    processed_stock_data = []
    for ticker in tickers:
        # Download data for a single ticker
        stock_data_ticker = yf.download(ticker, start=start_date, end=end_date)
        if stock_data_ticker.empty:
            continue

        # Ensure 'Date' is in the columns and it's a datetime object
        stock_data_ticker = stock_data_ticker.reset_index()
        stock_data_ticker['Date'] = pd.to_datetime(stock_data_ticker['Date'])
        
        # Sort by date
        stock_data_ticker = stock_data_ticker.sort_values(by='Date')
        
        # Calculate returns
        stock_data_ticker['returns'] = (stock_data_ticker['Close'] / stock_data_ticker['Close'].shift(1)) - 1
        stock_data_ticker['returns'] = stock_data_ticker['returns'].bfill()
        
        # Calculate volatility
        stock_data_ticker['volatility'] = stock_data_ticker['returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Add ticker column
        stock_data_ticker['ticker'] = ticker
        processed_stock_data.append(stock_data_ticker)

    if not processed_stock_data:
        print("Could not download or process any stock data.")
        exit()
        
    stock_data = pd.concat(processed_stock_data)

    # 4. Merge features with future volatility
    merged_data = []
    for index, row in features_df.iterrows():
        call_date = row['date']
        future_volatility_period = stock_data[
            (stock_data['ticker'] == row['ticker']) &
            (stock_data['Date'] > call_date) &
            (stock_data['Date'] <= call_date + pd.Timedelta(days=90))
        ]
        
        if not future_volatility_period.empty:
            avg_future_volatility = future_volatility_period['volatility'].mean()
            
            new_row = row.to_dict()
            new_row['avg_future_volatility'] = avg_future_volatility
            merged_data.append(new_row)

    final_df = pd.DataFrame(merged_data)
    
    if not final_df.empty:
        # 5. Correlation Analysis
        # Define feature columns dynamically
        feature_columns = [col for col in final_df.columns if col not in ['ticker', 'date', 'speaker', 'avg_future_volatility']]
        correlation_columns = feature_columns + ['avg_future_volatility']
        
        correlation = final_df[correlation_columns].corr()
        
        print("Correlation Matrix:")
        print(correlation)
        
        print("\nCorrelation with future volatility:")
        print(correlation['avg_future_volatility'].sort_values(ascending=False))
    else:
        print("Could not merge features with volatility data.")
