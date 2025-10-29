# analysis/features/core.py (No NLTK Tokenizer)
import textstat
import pandas as pd
import nltk
import os
import sys
import re  # <-- Import the regular expression library

# --- Robust Path Setup to find and configure NLTK ---
# We still need this for the sentiment analyzer
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    project_root = os.getcwd()

from analysis.helpers import configure_nltk_path
configure_nltk_path() # Configure NLTK path BEFORE using its modules
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def calculate_core_linguistic_features(data):
    """Calculates "Core" linguistic features from standard filings."""
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    sid = SentimentIntensityAnalyzer()
    features = []
    
    generalizing_words = ["generally", "typically", "fundamentals", "usually", "normally", "overall"]
    self_reference_words = ["i", "we", "my", "our", "mine", "ours"]

    for entry in data:
        text = entry.get('text', '')
        if not text: continue
        
        # --- REPLACEMENT FOR NLTK TOKENIZER ---
        # This is robust and has no external dependencies.
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens: continue

        # Features that DO NOT depend on the tokenizer change
        complexity_score = textstat.flesch_kincaid_grade(text)
        sentiment_score = sid.polarity_scores(text)['compound']
        
        # Features that now use the new tokenizer
        generalizing_score = sum(1 for word in tokens if word in generalizing_words) / len(tokens)
        self_reference_score = sum(1 for word in tokens if word in self_reference_words) / len(tokens)
        
        # --- FEATURE REMOVED ---
        # The Tense Analysis feature is removed as it requires nltk.pos_tag,
        # which depends on the punkt tokenizer.
        
        features.append({
            'ticker': entry.get('ticker'), 'date': entry.get('date'),
            'speaker': entry.get('speaker'), 'complexity_score': complexity_score,
            'sentiment_score': sentiment_score, 'generalizing_score': generalizing_score,
            'self_reference_score': self_reference_score
            # 'future_tense_ratio' and 'past_tense_ratio' are removed
        })
    return pd.DataFrame(features)