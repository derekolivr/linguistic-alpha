import pandas as pd
import yfinance as yf
from datetime import timedelta

# Assuming the feature extractors and data loader are in the analysis directory
from analysis.features.core import calculate_core_linguistic_features
from analysis.features.mda import calculate_mda_features
from analysis.features.risk_factors import calculate_risk_keyword_density
from analysis.data_loader import download_and_process_transcripts

def get_next_quarter_performance(ticker, date):
    """
    Fetches the stock's return and volatility for the 90-day period after a given date.
    """
    start_date = pd.to_datetime(date)
    end_date = start_date + timedelta(days=90)
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if stock_data.empty:
            return None, None
            
        # Calculate next quarter's return
        start_price = stock_data['Adj Close'].iloc[0]
        end_price = stock_data['Adj Close'].iloc[-1]
        next_quarter_return = (end_price - start_price) / start_price
        
        # Calculate next quarter's volatility (standard deviation of daily returns)
        stock_data['daily_return'] = stock_data['Adj Close'].pct_change()
        next_quarter_volatility = stock_data['daily_return'].std()
        
        # Ensure the return value is a scalar float, not a Series
        if isinstance(next_quarter_return, pd.Series):
            next_quarter_return = next_quarter_return.iloc[0]
            
        return next_quarter_return, next_quarter_volatility
    except Exception as e:
        print(f"Could not fetch stock data for {ticker}: {e}")
        return None, None

def run_transcript_feature_engineering():
    """
    Main function to run the transcript feature engineering pipeline.
    """
    # 1. Load Transcripts
    transcripts = download_and_process_transcripts()
    if not transcripts:
        print("No transcripts to process.")
        return
    
    # 2. Calculate Linguistic Features
    # We'll apply a selection of features from existing modules that are relevant to earnings calls.
    core_features = calculate_core_linguistic_features(transcripts)
    mda_features = calculate_mda_features(transcripts) # Forward-looking statements are very relevant
    risk_features = calculate_risk_keyword_density(transcripts) # Risk language is also key
    
    # 3. Merge Linguistic Features
    # We'll merge these feature sets together on ticker and date.
    merged_features = pd.merge(core_features, mda_features, on=['ticker', 'date'])
    merged_features = pd.merge(merged_features, risk_features, on=['ticker', 'date'])
    
    # 4. Calculate Z-Scores for Normalization
    linguistic_cols = [col for col in merged_features.columns if col not in ['ticker', 'date', 'speaker']]
    
    # Group by ticker and calculate z-score for each feature
    for col in linguistic_cols:
        # The transform function applies a function to each group and returns a series aligned with the original index
        merged_features[f'{col}_zscore'] = merged_features.groupby('ticker')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    # 5. Create a Composite Risk Score
    # This score combines several risk-related z-scores into a single metric.
    # We use -sentiment_score so that lower-than-average sentiment increases the risk.
    risk_components = [
        'complexity_score_zscore',
        'risk_keyword_density_zscore',
    ]
    # Ensure sentiment score z-score is present before trying to negate it
    if 'sentiment_score_zscore' in merged_features.columns:
        merged_features['neg_sentiment_zscore'] = -merged_features['sentiment_score_zscore']
        risk_components.append('neg_sentiment_zscore')

    merged_features['composite_risk_score'] = merged_features[risk_components].mean(axis=1)

    # Also calculate the z-score of the composite score itself for trend analysis
    merged_features['composite_risk_score_zscore'] = merged_features.groupby('ticker')['composite_risk_score'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # 6. Integrate Stock Performance Data (Efficient Batch Method)
    print("Downloading all required stock data in a single batch...")
    
    # Determine the date range for all stock data needed
    merged_features['date'] = pd.to_datetime(merged_features['date'])
    min_date = merged_features['date'].min()
    max_date = merged_features['date'].max() + timedelta(days=90)
    
    # Get unique tickers
    tickers = merged_features['ticker'].unique().tolist()
    
    # Download all stock data in one go
    try:
        all_stock_data = yf.download(tickers, start=min_date, end=max_date, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"Failed to download bulk stock data: {e}")
        return

    def calculate_performance(row):
        ticker = row['ticker']
        start_date = row['date']
        end_date = start_date + timedelta(days=90)
        
        # Slice the relevant data from the downloaded batch
        stock_slice = all_stock_data.loc[start_date:end_date]
        
        # Check if the slice is empty or the ticker data is missing
        if stock_slice.empty or ('Adj Close', ticker) not in stock_slice.columns:
            return pd.Series([None, None], index=['next_quarter_return', 'next_quarter_volatility'])
            
        adj_close = stock_slice['Adj Close'][ticker].dropna()

        if len(adj_close) < 2:
            return pd.Series([None, None], index=['next_quarter_return', 'next_quarter_volatility'])

        # Calculate return and volatility
        start_price = adj_close.iloc[0]
        end_price = adj_close.iloc[-1]
        next_quarter_return = (end_price - start_price) / start_price
        
        daily_return = adj_close.pct_change()
        next_quarter_volatility = daily_return.std()
        
        return pd.Series([next_quarter_return, next_quarter_volatility], index=['next_quarter_return', 'next_quarter_volatility'])

    print("Calculating performance metrics for each transcript...")
    performance_df = merged_features.apply(calculate_performance, axis=1)

    # 7. Combine all data
    final_df = pd.concat([merged_features.reset_index(drop=True), performance_df.reset_index(drop=True)], axis=1)

    # 8. Create Classification Targets
    # Return Class: 1 if return is positive, 0 otherwise
    final_df['return_class'] = (final_df['next_quarter_return'] > 0).astype(int)

    # Volatility Class: 1 if volatility is above the stock's historical median, 0 otherwise
    # Need to handle potential NaNs in the volatility column before calculating median
    final_df['volatility_class'] = final_df.groupby('ticker')['next_quarter_volatility'].transform(
        lambda x: (x > x.median()).astype(int) if x.notna().any() else x
    )
    
    # Save to CSV
    output_path = 'output/transcript_features_with_performance.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")

if __name__ == '__main__':
    run_transcript_feature_engineering()
