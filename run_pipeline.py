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
# NOTE: The parser import is commented out, as we are not using it.
# from parser.parserr import parse_all_downloads

from analysis.helpers import load_mock_data, configure_nltk_path
configure_nltk_path() # Configure NLTK path before it's used
from analysis.features.core import calculate_core_linguistic_features
from analysis.features.mda import calculate_mda_features
from analysis.features.risk_factors import (
    calculate_risk_keyword_density,
    calculate_risk_specificity,
    calculate_risk_factor_change
)
from analysis.transcript_feature_engineering import run_transcript_feature_engineering

def main():
    """
    Orchestrates the analysis pipeline using pre-existing mock data
    from 'data/parsed_filings.json'. SKIPS the live parsing step.
    """
    print("--- Starting Linguistic Alpha Pipeline (USING MOCK DATA) ---")

    # --- Step 1: Parse Raw Data (SKIPPED) ---
    print("\n[Phase 1/4] Parsing raw filings... (SKIPPED)")
    # The line below is commented out.
    # parse_all_downloads(downloads_root="scraper/downloads", out_dir="data")

    # --- Step 2: Load and Filter Data ---
    print("\n[Phase 2/4] Loading and filtering data by section...")
    try:
        all_data = load_mock_data('parsed_filings.json', project_root)
    except FileNotFoundError:
        print("\n[!] ERROR: 'data/parsed_filings.json' not found.")
        print("    -> Please ensure your mock data file is created and named correctly.")
        return # Exit the script
    
    mda_data = [d for d in all_data if d.get("section") == "MD&A"]
    risk_data = [d for d in all_data if d.get("section") == "Risk_Factors"]

    # --- Step 3: Calculate Features Efficiently ---
    print("\n[Phase 3/4] Calculating linguistic features...")
    
    # Calculate and merge features for MD&A
    mda_base_features = calculate_core_linguistic_features(mda_data)
    mda_spec_features = calculate_mda_features(mda_data)
    mda_full_features = pd.merge(mda_base_features, mda_spec_features, on=['ticker', 'date'], how='outer')
    mda_full_features['date'] = pd.to_datetime(mda_full_features['date'])
    mda_full_features = mda_full_features.add_suffix('_mda').rename(columns={'ticker_mda': 'ticker', 'date_mda': 'date'})
    
    # Calculate and merge features for Risk Factors
    risk_base_features = calculate_core_linguistic_features(risk_data)
    risk_density_df = calculate_risk_keyword_density(risk_data)
    risk_spec_df = calculate_risk_specificity(risk_data)
    risk_change_df = calculate_risk_factor_change(risk_data)
    risk_full_features = pd.merge(risk_base_features, risk_density_df, on=['ticker', 'date'], how='outer')
    risk_full_features = pd.merge(risk_full_features, risk_spec_df, on=['ticker', 'date'], how='outer')
    risk_full_features['date'] = pd.to_datetime(risk_full_features['date'])
    risk_full_features = pd.merge(risk_full_features, risk_change_df, on=['ticker', 'date'], how='outer')
    risk_full_features = risk_full_features.add_suffix('_risk').rename(columns={'ticker_risk': 'ticker', 'date_risk': 'date'})

    # --- Step 4: Merge All Features and Save ---
    print("\n[Phase 4/4] Merging all features and saving final output...")
    final_features_df = pd.merge(mda_full_features, risk_full_features, on=['ticker', 'date'], how='inner')
    
    print("Creating cross-sectional features...")
    if not final_features_df.empty:
        final_features_df['tone_mismatch'] = final_features_df['sentiment_score_mda'] - final_features_df['risk_keyword_density_risk']
        final_features_df['complexity_delta'] = final_features_df['complexity_score_risk'] - final_features_df['complexity_score_mda']
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    final_features_df.to_csv(os.path.join(output_dir, "final_linguistic_features.csv"), index=False)

    print(f"Successfully saved {len(final_features_df)} records to 'output/final_linguistic_features.csv'")
    print("\n--- Pipeline Finished ---")

    print("\n--- Starting Earnings Transcript Analysis Pipeline ---")
    run_transcript_feature_engineering()
    print("--- Earnings Transcript Analysis Finished ---")


if __name__ == "__main__":
    main()