import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import pandas as pd
import os

def download_nltk_resources():
    """Download necessary NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
    try:
        nltk.data.find('taggers/universal_tagset')
    except LookupError:
        nltk.download('universal_tagset')

def find_project_root(marker='requirements.txt'):
    """Find the project root by looking for a marker file."""
    try:
        # This works when running as a .py script
        start_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments like Jupyter
        start_dir = os.getcwd()

    current_dir = start_dir
    while True:
        if marker in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Project root marker '{marker}' not found from '{start_dir}'.")
        current_dir = parent_dir

def load_mock_data(filepath=None):
    """Load mock data from a JSON file."""
    if filepath is None:
        project_root = find_project_root()
        filepath = os.path.join(project_root, 'data', 'mock_data.json')
        
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_linguistic_features(data):
    """
    Calculate linguistic features for each text entry in the data.
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
        
        # 2. Sentiment Score
        sentiment_score = sid.polarity_scores(text)['compound']

        # 3. Generalizing Language
        generalizing_score = sum(1 for word in tokens if word in generalizing_words) / len(tokens) if tokens else 0
        
        # 4. Self-Reference
        self_reference_score = sum(1 for word in tokens if word in self_reference_words) / len(tokens) if tokens else 0
        
        # 5. Tense Analysis
        tagged_words = nltk.pos_tag(tokens, tagset='universal')
        future_tense_verbs = [word for word, tag in tagged_words if tag in ['VERB']] # Modal verbs are tagged as VERB with universal tagset
        past_tense_verbs = [word for word, tag in tagged_words if tag in ['VERB']]
        
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

if __name__ == "__main__":
    download_nltk_resources()
    mock_data = load_mock_data()
    features_df = calculate_linguistic_features(mock_data)
    print(features_df)
