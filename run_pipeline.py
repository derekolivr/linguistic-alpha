import pandas as pd
import os
import sys

# --- Robust Path Setup ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import All Necessary Modules ---
from analysis.helpers import configure_nltk_path
configure_nltk_path() # Configure NLTK path before it's used
from analysis.transcript_feature_engineering import run_transcript_feature_engineering
from analysis.model_training import train_all_models
from analysis.backtest import run_backtest

def main():
    """
    Orchestrates the end-to-end earnings transcript analysis pipeline.
    This pipeline fetches real-world data, engineers features, trains
    predictive models, and evaluates their performance.
    """
    print("--- Starting Earnings Transcript Analysis Pipeline ---")
    run_transcript_feature_engineering()
    print("--- Feature Engineering Complete ---")

    print("\n--- Training Predictive Models ---")
    train_all_models()
    print("--- Model Training Complete ---")

    print("\n--- Running Backtest on Hold-out Data ---")
    run_backtest()
    print("--- Backtest Complete ---")

    print("\n--- Pipeline Finished ---")
    print("You can now view the results in the Streamlit dashboard.")

if __name__ == "__main__":
    main()