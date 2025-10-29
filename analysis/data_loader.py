import pandas as pd
from datasets import load_dataset

def download_and_process_transcripts():
    """
    Downloads, filters, and processes S&P 500 earnings call transcripts.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("kurry/sp500_earnings_transcripts", split='train')
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)
    
    # Define the tickers and date range
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    start_year = 2023
    end_year = 2024
    
    # Filter for the target tickers and years
    filtered_df = df[
        (df['symbol'].isin(target_tickers)) &
        (df['year'] >= start_year) &
        (df['year'] <= end_year)
    ]
    
    # Standardize the column names and format
    # The plan requires 'ticker', 'date', and 'text'
    # The dataset provides 'symbol', 'date', and 'content'
    processed_data = []
    for _, row in filtered_df.iterrows():
        processed_data.append({
            'ticker': row['symbol'],
            'date': row['date'],
            'text': row['content']
        })
        
    return processed_data

if __name__ == '__main__':
    # Example of how to run the function and see the output
    transcripts = download_and_process_transcripts()
    if transcripts:
        print(f"Successfully downloaded and processed {len(transcripts)} transcripts.")
        print("Sample transcript:")
        print(transcripts[0])
    else:
        print("No transcripts found for the specified criteria.")
