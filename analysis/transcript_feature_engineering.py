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
    
    # 5. Integrate Stock Performance Data
    performance_data = []
    for _, row in merged_features.iterrows():
        ret, vol = get_next_quarter_performance(row['ticker'], row['date'])
        performance_data.append({
            'next_quarter_return': ret,
            'next_quarter_volatility': vol
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 6. Combine all data and save
    final_df = pd.concat([merged_features.reset_index(drop=True), performance_df.reset_index(drop=True)], axis=1)
    
    # Save to CSV
    output_path = 'output/transcript_features_with_performance.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")

if __name__ == '__main__':
    run_transcript_feature_engineering()
