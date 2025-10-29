# analysis/helpers.py
import nltk
import os
import sys
import json

def configure_nltk_path():
    """
    Tells NLTK to look for data in the project's local 'nltk_data' folder.
    This is essential for making the project self-contained and avoiding
    system-level pathing issues.
    """
    # This assumes helpers.py is in 'root/analysis/helpers.py'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_nltk_path = os.path.join(project_root, 'nltk_data')
    
    # Prepend the local path to NLTK's search path. It will look here first.
    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)

def find_project_root(marker='requirements.txt'):
    # This function remains useful for other scripts
    start_dir = os.getcwd()
    current_dir = start_dir
    while True:
        if marker in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Project root marker '{marker}' not found.")
        current_dir = parent_dir

def load_mock_data(filename, project_root):
    """Loads JSON data from the 'data' directory."""
    filepath = os.path.join(project_root, 'data', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)