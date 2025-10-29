import json
import os
import nltk

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

def load_mock_data(filename, project_root):
    """Load mock data from a JSON file within the project structure."""
    filepath = os.path.join(project_root, 'data', filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mock data file not found at: {filepath}. Please ensure it exists.")
    with open(filepath, 'r') as f:
        return json.load(f)
