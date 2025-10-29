# analysis/features/risk_factors.py
import pandas as pd
import re
from difflib import SequenceMatcher

# This dictionary can be expanded over time
RISK_KEYWORDS = [
    'adverse', 'risk', 'uncertainty', 'depend', 'contingent', 'could',
    'may', 'might', 'volatile', 'fluctuate', 'materially', 'violation',
    'impairment', 'decline', 'challenging', 'significant'
]

def calculate_risk_keyword_density(data):
    """Calculates the density of specific risk-related keywords."""
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    features = []
    for entry in data:
        text = entry.get('text', '').lower()
        if not text:
            continue
        
        tokens = re.findall(r'\b\w+\b', text)
        if not tokens:
            continue
            
        risk_word_count = sum(1 for word in tokens if word in RISK_KEYWORDS)
        density = risk_word_count / len(tokens)
        
        features.append({
            'ticker': entry.get('ticker'),
            'date': entry.get('date'),
            'risk_keyword_density': density
        })
        
    return pd.DataFrame(features)

def calculate_risk_specificity(data):
    """Calculates the density of numbers, a proxy for specificity."""
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    features = []
    for entry in data:
        text = entry.get('text', '')
        if not text:
            continue
            
        digits = sum(c.isdigit() for c in text)
        specificity = digits / len(text) if len(text) > 0 else 0
        
        features.append({
            'ticker': entry.get('ticker'),
            'date': entry.get('date'),
            'risk_specificity_score': specificity
        })
        
    return pd.DataFrame(features)

def calculate_risk_factor_change(data):
    """
    Calculates the change in risk factor text from the previous quarter.
    This is an advanced feature and requires data to be sorted by date.
    """
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame() # Cannot calculate change without historical data

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date'])
    
    # Use SequenceMatcher for a simple text similarity score
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Group by ticker and compare with the previous period's text
    df['previous_text'] = df.groupby('ticker')['text'].shift(1)
    
    # Calculate similarity and then change (1 - similarity)
    # Fill NaN for the very first filing of a company
    df['similarity_score'] = df.apply(
        lambda row: similarity(row['text'], row['previous_text']) if pd.notna(row['previous_text']) else None,
        axis=1
    )
    df['risk_text_change_score'] = 1 - df['similarity_score']
    
    return df[['ticker', 'date', 'risk_text_change_score']].dropna()