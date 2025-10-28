import pandas as pd
import yfinance as yf
import numpy as np
from .feature_engineering import find_project_root, load_mock_data, calculate_core_linguistic_features

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
    # 1. Load linguistic features
    project_root = find_project_root()
    mock_data = load_mock_data('mock_filings.json', project_root)
    features_df = calculate_core_linguistic_features(mock_data)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # 2. Get stock data
    tickers = features_df['ticker'].unique().tolist()
    start_date = features_df['date'].min() - pd.Timedelta(days=1)
    # We need data for 90 days after the last earnings call to calculate future volatility
    end_date = features_df['date'].max() + pd.Timedelta(days=91) 

    all_stock_data = []
    for ticker in tickers:
        stock_data_ticker = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data_ticker.empty:
            stock_data_ticker['ticker'] = ticker
            all_stock_data.append(stock_data_ticker)

    if all_stock_data:
        stock_data = pd.concat(all_stock_data)
    else:
        print("Could not download any stock data.")
        exit()

    # 3. Calculate volatility
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)

    if not stock_data.empty:
        # Ensure 'Date' is in datetime format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Sort values to ensure correct rolling calculations
        stock_data = stock_data.sort_values(by=['ticker', 'Date'])
        
        # --- Data Type Test ---
        returns_calculation = stock_data.groupby('ticker')['Close'].pct_change()
        print(f"--- Data Type Inspection ---")
        print(f"Type of calculated returns: {type(returns_calculation)}")
        print(f"Shape: {getattr(returns_calculation, 'shape', 'N/A')}")
        if hasattr(returns_calculation, 'columns'):
            print(f"Columns: {returns_calculation.columns}")
        print(f"--------------------------\n")
        
        # Calculate returns for each ticker and reshape the result
        returns_calculation = stock_data.groupby('ticker')['Close'].pct_change()
        
        # The result of the above is a wide-format DataFrame. We stack it to turn it into a long-format Series.
        if isinstance(returns_calculation, pd.DataFrame):
            returns_calculation = returns_calculation.stack(level='Ticker').reset_index()
            returns_calculation.rename(columns={'level_0': 'Date', 'Close': 'returns'}, inplace=True)
            returns_calculation.set_index('Date', inplace=True)

        stock_data['returns'] = returns_calculation['returns']
        
        # Backfill the first NaN value for each ticker that arises from pct_change()
        stock_data['returns'] = stock_data.groupby('ticker')['returns'].transform(lambda x: x.bfill())

        # Calculate rolling volatility for each ticker
        stock_data['volatility'] = stock_data.groupby('ticker')['returns'].transform(lambda x: x.rolling(window=30).std()) * np.sqrt(252)

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
            correlation = final_df[['complexity_score', 'sentiment_score', 'generalizing_score', 'self_reference_score', 'future_tense_ratio', 'past_tense_ratio', 'avg_future_volatility']].corr()
            
            print("Correlation Matrix:")
            print(correlation)
            
            print("\nCorrelation with future volatility:")
            print(correlation['avg_future_volatility'].sort_values(ascending=False))
        else:
            print("Could not merge features with volatility data.")
