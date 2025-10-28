import nltk
import textstat
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
