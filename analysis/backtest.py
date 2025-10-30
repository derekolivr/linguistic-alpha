import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def run_backtest():
    """
    Loads the trained models and evaluates their performance on the 2024 hold-out set.
    """
    # --- Load Data ---
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # --- Temporal Split: Get Test Set ---
    test_df = df[df['date'].dt.year == 2024].copy()
    
    if test_df.empty:
        print("No data available for 2024 to run the backtest.")
        return

    # --- Feature Selection ---
    feature_cols = [col for col in df.columns if col.endswith('_zscore')]
    X_test = test_df[feature_cols].fillna(0)

    # --- Evaluate Return Model ---
    print("\n--- Evaluating Return Prediction Model ---")
    return_model_path = 'output/return_classifier.joblib'
    if os.path.exists(return_model_path):
        return_model = joblib.load(return_model_path)
        y_test_return = test_df['return_class']
        
        if not y_test_return.empty:
            predictions = return_model.predict(X_test)
            print("Classification Report (Returns):")
            print(classification_report(y_test_return, predictions, zero_division=0))
            print(f"AUC Score (Returns): {roc_auc_score(y_test_return, return_model.predict_proba(X_test)[:, 1]):.4f}")
        else:
            print("No return data to evaluate.")
    else:
        print("Return model not found. Please train the model first.")

    # --- Evaluate Volatility Model ---
    print("\n--- Evaluating Volatility Prediction Model ---")
    volatility_model_path = 'output/volatility_classifier.joblib'
    if os.path.exists(volatility_model_path):
        volatility_model = joblib.load(volatility_model_path)
        y_test_vol = test_df['volatility_class']
        
        if not y_test_vol.empty:
            predictions = volatility_model.predict(X_test)
            print("Classification Report (Volatility):")
            print(classification_report(y_test_vol, predictions, zero_division=0))
            print(f"AUC Score (Volatility): {roc_auc_score(y_test_vol, volatility_model.predict_proba(X_test)[:, 1]):.4f}")
        else:
            print("No volatility data to evaluate.")
    else:
        print("Volatility model not found. Please train the model first.")

if __name__ == '__main__':
    run_backtest()
