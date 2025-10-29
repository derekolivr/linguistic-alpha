import os
import pandas as pd
from analysis.helpers import find_project_root, load_mock_data, download_nltk_resources
from analysis.features.core import calculate_core_linguistic_features
from analysis.features.catalyst import calculate_catalyst_score
from analysis.features.mda import calculate_mda_features

def main():
    """
    Main function to run the entire linguistic feature engineering pipeline.
    """
    # --- Setup ---
    print("Setting up the environment...")
    download_nltk_resources()
    project_root = find_project_root()
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Data ---
    print("Loading mock data...")
    try:
        core_data = load_mock_data('mock_filings.json', project_root)
        catalyst_data = load_mock_data('mock_events.json', project_root)
        mda_data = load_mock_data('mock_mda.json', project_root)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- Feature Engineering ---
    print("Calculating core linguistic features...")
    core_features_df = calculate_core_linguistic_features(core_data)
    
    print("Calculating catalyst event scores...")
    catalyst_features_df = calculate_catalyst_score(catalyst_data)
    
    print("Calculating MD&A features...")
    mda_features_df = calculate_mda_features(mda_data)

    # --- Save Outputs ---
    print("Saving feature dataframes to output directory...")
    core_features_df.to_csv(os.path.join(output_dir, 'core_features.csv'), index=False)
    catalyst_features_df.to_csv(os.path.join(output_dir, 'catalyst_features.csv'), index=False)
    mda_features_df.to_csv(os.path.join(output_dir, 'mda_features.csv'), index=False)

    print("\nPipeline finished successfully!")
    print(f"Output files saved in: {output_dir}")

    # --- Display Sample Outputs ---
    print("\n--- Sample Core Features ---")
    print(core_features_df.head())
    print("\n--- Sample Catalyst Features ---")
    print(catalyst_features_df.head())
    print("\n--- Sample MD&A Features ---")
    print(mda_features_df.head())

if __name__ == "__main__":
    main()
