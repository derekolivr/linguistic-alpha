import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

def run_backtest():
    """
    Performs a time-series cross-validation (walk-forward analysis) to get a
    robust measure of model performance.
    """
    # --- Load Data ---
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    df = pd.read_csv(file_path).dropna(subset=['return_class', 'volatility_class'])
    df['date'] = pd.to_datetime(df['date'])
    
    TEST_YEARS = [2023, 2024]
    all_results = []

    for year in TEST_YEARS:
        print(f"--- Backtesting on Year: {year} ---")
        
        # --- Temporal Split for this fold ---
        train_df = df[df['date'].dt.year < year]
        test_df = df[df['date'].dt.year == year].copy()

        if test_df.empty or train_df.empty:
            print(f"Skipping year {year} due to lack of data.")
            continue

        # --- Feature Selection ---
        feature_cols = [col for col in df.columns if col.endswith('_zscore')]
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)

        # --- Train and Evaluate Return Model for this fold ---
        y_train_return = train_df['return_class']
        y_test_return = test_df['return_class']
        scale_pos_weight_return = (y_train_return == 0).sum() / (y_train_return == 1).sum()
        estimators_return = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_return))
        ]
        stacking_classifier_return = StackingClassifier(estimators=estimators_return, final_estimator=LogisticRegression())
        return_model = stacking_classifier_return.fit(X_train, y_train_return)
        
        test_df['return_pred'] = return_model.predict(X_test)
        test_df['return_proba'] = return_model.predict_proba(X_test)[:, 1]

        # --- Train and Evaluate Volatility Model for this fold ---
        y_train_vol = train_df['volatility_class']
        y_test_vol = test_df['volatility_class']
        scale_pos_weight_vol = (y_train_vol == 0).sum() / (y_train_vol == 1).sum()
        estimators_vol = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_vol))
        ]
        stacking_classifier_vol = StackingClassifier(estimators=estimators_vol, final_estimator=LogisticRegression())
        volatility_model = stacking_classifier_vol.fit(X_train, y_train_vol)
        
        test_df['volatility_pred'] = volatility_model.predict(X_test)
        test_df['volatility_proba'] = volatility_model.predict_proba(X_test)[:, 1]
        
        all_results.append(test_df)

    # --- Aggregate and Save Results ---
    if not all_results:
        print("Backtest could not be completed.")
        return
        
    results_df = pd.concat(all_results)
    results_df.to_csv('output/backtest_results.csv', index=False)
    print("\n--- Aggregated Backtest Results (2023-2024) ---")

    # --- Print Aggregated Return Model Performance ---
    print("\n--- Evaluating Return Prediction Model ---")
    y_true_return = results_df['return_class']
    y_pred_return = results_df['return_pred']
    print(classification_report(y_true_return, y_pred_return))
    print(f"Overall Accuracy: {accuracy_score(y_true_return, y_pred_return):.4f}")
    print(f"Overall AUC: {roc_auc_score(y_true_return, results_df['return_proba']):.4f}")

    # --- Print Aggregated Volatility Model Performance ---
    print("\n--- Evaluating Volatility Prediction Model ---")
    y_true_vol = results_df['volatility_class']
    y_pred_vol = results_df['volatility_pred']
    print(classification_report(y_true_vol, y_pred_vol))
    print(f"Overall Accuracy: {accuracy_score(y_true_vol, y_pred_vol):.4f}")
    print(f"Overall AUC: {roc_auc_score(y_true_vol, results_df['volatility_proba']):.4f}")

if __name__ == '__main__':
    run_backtest()
