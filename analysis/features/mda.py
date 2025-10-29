import nltk
import textstat
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def calculate_mda_features(data):
    """
    Calculates features from MD&A sections, focusing on forward-looking statements
    and risk disclosures.
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
        tokens = nltk.word_tokenize(text.lower())

        # 1. Forward-Looking Statement Ratio
        forward_looking_ratio = sum(1 for word in tokens if word in forward_looking_words) / len(tokens) if tokens else 0

        # 2. Risk Factor Density
        risk_factor_density = sum(1 for word in tokens if word in risk_words) / len(tokens) if tokens else 0

        # 3. Quantitative Disclosure Ratio (as a proxy for concreteness)
        quantitative_ratio = sum(1 for word in tokens if word.isdigit()) / len(tokens) if tokens else 0

        # Re-use existing metrics
        complexity_score = textstat.flesch_kincaid_grade(text)
        sentiment_score = sid.polarity_scores(text)['compound']

        feature_entry = {
            'ticker': entry['ticker'],
            'date': entry['date'],
            'complexity_score': complexity_score,
            'sentiment_score': sentiment_score,
            'forward_looking_ratio': forward_looking_ratio,
            'risk_factor_density': risk_factor_density,
            'quantitative_ratio': quantitative_ratio
        }
        features.append(feature_entry)

    return pd.DataFrame(features)
