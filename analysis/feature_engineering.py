# %% [markdown]
# # Psycholinguistic Feature Engineering
# 
# This script is responsible for feature engineering. It takes raw text data from corporate communications (like earnings calls or SEC filings) and calculates a vector of psycholinguistic features. These features are designed to capture the tone, style, and potential subtext of the communication, which can then be used to predict financial market outcomes like stock volatility.
# 
# The script is divided into two main feature calculation functions:
# 1.  `calculate_core_linguistic_features`: Analyzes the general communication style from standard filings.
# 2.  `calculate_catalyst_score`: Analyzes severity and intent from crucial, event-driven texts.

# %%
import json
import nltk
import textstat
import pandas as pd
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# %% [markdown]
# ## Setup and Helper Functions
# 
# These functions handle the necessary setup, such as downloading language resources and locating the project's data files in a robust way.

# %%
def download_nltk_resources():
    """Download necessary NLTK resources if not already present."""
    resources = {
        'tokenizers/punkt': 'punkt',
        'sentiment/vader_lexicon.zip': 'vader_lexicon',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    for path, resource_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK resource '{resource_id}' not found. Downloading...")
            nltk.download(resource_id)

# %%
def find_project_root(marker='requirements.txt'):
    """Find the project root by looking for a marker file."""
    try:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start_dir = os.getcwd()

    current_dir = start_dir
    while True:
        if marker in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # Fallback for when the script is not in a project structure
            print(f"Warning: Project root marker '{marker}' not found. Using current directory '{start_dir}'.")
            return start_dir
        current_dir = parent_dir

# %%
def load_mock_data(filename, project_root):
    """Load mock data from a JSON file within the project structure."""
    filepath = os.path.join(project_root, 'data', filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mock data file not found at: {filepath}. Please ensure it exists.")
    with open(filepath, 'r') as f:
        return json.load(f)

# %% [markdown]
# ## 1. Core Linguistic Features
# 
# This function calculates "Core" linguistic features from standard, recurring filings like 10-Ks and 10-Qs. The goal is to profile the *general communication style* of a company over time. These features are not designed to capture sudden events but rather the underlying tendencies in their language.

# %%
def calculate_core_linguistic_features(data):
    """
    Calculates "Core" linguistic features from standard filings (10-Ks, 10-Qs).
    This function profiles the general communication style of a company.
    """
    sid = SentimentIntensityAnalyzer()
    features = []
    
    generalizing_words = ["generally", "typically", "fundamentals", "usually", "normally", "overall"]
    self_reference_words = ["i", "we", "my", "our", "mine", "ours"]

    for entry in data:
        text = entry['text']
        tokens = nltk.word_tokenize(text.lower())
        
        # 1. Sentence Complexity (Flesch-Kincaid Grade Level)
        complexity_score = textstat.flesch_kincaid_grade(text)
        
        # 2. General Sentiment Score
        sentiment_score = sid.polarity_scores(text)['compound']

        # 3. Generalizing Language Density
        generalizing_score = sum(1 for word in tokens if word in generalizing_words) / len(tokens) if tokens else 0
        
        # 4. Self-Reference Density
        self_reference_score = sum(1 for word in tokens if word in self_reference_words) / len(tokens) if tokens else 0
        
        # 5. Tense Analysis (Corrected for accuracy)
        tagged_words = nltk.pos_tag(tokens)
        future_tense_verbs = [word for word, tag in tagged_words if tag == 'MD']  # MD = Modal verb (will, shall, etc.)
        past_tense_verbs = [word for word, tag in tagged_words if tag == 'VBD']  # VBD = Verb, past tense
        
        future_tense_ratio = len(future_tense_verbs) / len(tokens) if tokens else 0
        past_tense_ratio = len(past_tense_verbs) / len(tokens) if tokens else 0

        feature_entry = {
            'ticker': entry['ticker'],
            'date': entry['date'],
            'speaker': entry['speaker'],
            'complexity_score': complexity_score,
            'sentiment_score': sentiment_score,
            'generalizing_score': generalizing_score,
            'self_reference_score': self_reference_score,
            'future_tense_ratio': future_tense_ratio,
            'past_tense_ratio': past_tense_ratio
        }
        features.append(feature_entry)
        
    return pd.DataFrame(features)

# %% [markdown]
# ## 2. Catalyst Event Score
# 
# This function calculates a "Catalyst" score from crucial, event-driven texts, such as a short-seller report or a company's rebuttal. Unlike the core features, this analysis focuses on **severity and intent**, using specialized dictionaries to detect signs of financial distress or fraudulent language.

# %%
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

# %% [markdown]
# ## Main Execution Block
# 
# This block demonstrates how to use the functions in this script. It loads the mock data for both core filings and catalyst events, calculates the respective features, and prints the resulting DataFrames.

# %%
if __name__ == "__main__":
    # --- Setup ---
    download_nltk_resources()
    try:
        project_root = find_project_root()

        # --- Part 1: Analyze "Core" Company Filings ---
        print("--- Analyzing Core Signals (Company Filings) ---")
        core_mock_data = load_mock_data('mock_filings.json', project_root)
        core_features_df = calculate_core_linguistic_features(core_mock_data)
        print("Core Linguistic Features DataFrame:")
        print(core_features_df)
        print("\n" + "="*50 + "\n")

        # --- Part 2: Analyze "Catalyst" Crucial Events ---
        print("--- Analyzing Catalyst Signals (Crucial Events) ---")
        catalyst_mock_data = load_mock_data('mock_events.json', project_root)
        catalyst_features_df = calculate_catalyst_score(catalyst_mock_data)
        print("Catalyst Event Score DataFrame:")
        print(catalyst_features_df)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have a 'data' folder in your project root with 'mock_filings.json' and 'mock_events.json'.")