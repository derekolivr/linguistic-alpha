import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

def calculate_features_by_section(data):
    """
    Calculates linguistic features for SEC filing sections (MD&A and Risk Factors).
    """
    sid = SentimentIntensityAnalyzer()
    features = []

    forward_looking_words = [
        "will", "expect", "believe", "future", "outlook", "guidance", "project", "anticipate"
    ]
    risk_words = [
        "risk", "uncertain", "depend", "compete", "threat", "challenge", "could", "may"
    ]

    for entry in data:
        text = entry['text']
        section = entry['section']
        tokens = nltk.word_tokenize(text.lower())
        
        feature_entry = {
            'ticker': entry['ticker'],
            'date': entry['date'],
            'section': section
        }

        if section == "MD&A":
            # Features relevant to MD&A's narrative style
            feature_entry['sentiment_score'] = sid.polarity_scores(text)['compound']
            feature_entry['forward_looking_ratio'] = sum(1 for word in tokens if word in forward_looking_words) / len(tokens) if tokens else 0
            feature_entry['complexity_score'] = textstat.flesch_kincaid_grade(text)
        
        elif section == "Risk_Factors":
            # Feature focused on risk disclosure
            feature_entry['risk_factor_density'] = sum(1 for word in tokens if word in risk_words) / len(tokens) if tokens else 0

        features.append(feature_entry)

    # Convert to DataFrame and pivot the data
    df = pd.DataFrame(features)
    
    # Pivot the table to have one row per ticker-date, with feature columns for each section
    pivot_df = df.pivot_table(
        index=['ticker', 'date'], 
        columns='section', 
        values=[col for col in df.columns if col not in ['ticker', 'date', 'section']]
    ).reset_index()

    # Flatten the multi-level column headers
    pivot_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pivot_df.columns.values]
    pivot_df = pivot_df.rename(columns={'ticker_': 'ticker', 'date_': 'date'})
    
    return pivot_df
