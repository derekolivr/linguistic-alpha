# analysis/features/mda.py (No NLTK Tokenizer)
import pandas as pd
import nltk
import os
import sys
import re # <-- Import the regular expression library

# --- Robust Path Setup to find and configure NLTK ---
# Still needed for sentiment analysis
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    project_root = os.getcwd()

from analysis.helpers import configure_nltk_path
configure_nltk_path()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def calculate_mda_features(data):
    """Calculates features UNIQUE to MD&A sections."""
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
        
    features = []

    forward_looking_words = ["will", "expect", "believe", "future", "outlook", "guidance", "project", "anticipate"]
    positive_words = ["achieved", "growth", "strong", "record", "exceeded", "successful"]

    for entry in data:
        text = entry.get('text', '')
        if not text: continue
        
        # --- REPLACEMENT FOR NLTK TOKENIZER ---
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens: continue

        forward_looking_ratio = sum(1 for word in tokens if word in forward_looking_words) / len(tokens)
        quantitative_tokens = re.findall(r'\d+', text)
        quantitative_ratio = len(quantitative_tokens) / len(tokens)
        positive_tone_density = sum(1 for word in tokens if word in positive_words) / len(tokens)
        
        features.append({
            'ticker': entry.get('ticker'), 'date': entry.get('date'),
            'forward_looking_ratio': forward_looking_ratio,
            'quantitative_ratio': quantitative_ratio,
            'positive_tone_density': positive_tone_density
        })
    return pd.DataFrame(features)