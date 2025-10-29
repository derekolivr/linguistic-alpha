import nltk
import pandas as pd

def calculate_catalyst_score(data):
    """
    Calculates "Catalyst" event scores from crucial, event-driven texts.
    This function analyzes severity and intent, not general style.
    """
    features = []

    # Specialized dictionaries for financial context
    financial_negative_words = [
        'loss', 'decline', 'impairment', 'volatile', 'risk', 'uncertainty', 'liability', 
        'default', 'restatement', 'investigation', 'downside', 'weakness', 'claim', 'alleges'
    ]
    fraud_keywords = [
        'fraud', 'sham', 'mislead', 'deception', 'undisclosed', 'unaccounted', 
        'manipulation', 'misstatement', 'illegal', 'scandal'
    ]

    for entry in data:
        text = entry['text']
        event_type = entry['event_type']
        tokens = nltk.word_tokenize(text.lower())
        
        # 1. Financial-Specific Negative Sentiment
        fin_neg_score = sum(1 for word in tokens if word in financial_negative_words) / len(tokens) if tokens else 0

        # 2. Fraud Keyword Density
        fraud_score = sum(1 for word in tokens if word in fraud_keywords) / len(tokens) if tokens else 0

        # 3. Data Specificity (Credibility proxy)
        # Measures the density of numbers in the text.
        digits = sum(c.isdigit() for c in text)
        specificity_score = digits / len(text) if text else 0

        # Combine features into a final "Catalyst Score"
        # The weights here are a starting point and would be optimized in a full backtest.
        # A higher score indicates a more severe negative event.
        catalyst_score = (fin_neg_score * 0.4) + (fraud_score * 0.4) + (specificity_score * 0.2)
        
        # Event-Type Logic: A rebuttal is scored differently.
        # For this example, we simply note the type, but a real model could have different logic.
        if event_type == "Company_Rebuttal":
            # A good rebuttal should be specific and low on negative words.
            # So, a low score is better for the company.
            final_score_label = "Rebuttal_Severity_Score"
        else:
            final_score_label = "Attack_Score"

        feature_entry = {
            'ticker': entry['ticker'],
            'date': entry['date'],
            'event_type': event_type,
            'source': entry['source'],
            final_score_label: round(catalyst_score * 1000, 2), # Scale for readability
            'specificity_score': round(specificity_score, 4)
        }
        features.append(feature_entry)

    return pd.DataFrame(features)
