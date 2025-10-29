import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

# --- Define Preset Scenarios ---
PRESET_GROUPS = {
    "Big Tech (FAANG+)": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX'],
    "Semiconductors": ['NVDA', 'AMD', 'INTC', 'QCOM', 'AMAT', 'AVGO'],
    "Cloud & Software": ['MSFT', 'CRM', 'ADBE', 'INTU', 'NOW', 'ORCL'],
    "All Selected Companies": [
        'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'NFLX', 'AMD', 'CRM', 
        'ADBE', 'INTU', 'NOW', 'AMAT', 'CSCO'
    ]
}

def run_backtest():
    """
    Performs a time-series cross-validation for several preset groups of companies.
    """
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    df = pd.read_csv(file_path).dropna(subset=['return_class', 'volatility_class'])
    df['date'] = pd.to_datetime(df['date'])
    
    all_preset_results = []

    for group_name, tickers in PRESET_GROUPS.items():
        print(f"\n--- Running backtest for group: {group_name} ---")
        group_df = df[df['ticker'].isin(tickers)]
        
        # This is the same walk-forward logic as before, now applied per-group
        TEST_YEARS = [2023, 2024]
        for year in TEST_YEARS:
            train_df = group_df[group_df['date'].dt.year < year]
            test_df = group_df[group_df['date'].dt.year == year].copy()

            if test_df.empty or train_df.empty:
                continue

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
            
            # After predictions are made:
            test_df['preset_group'] = group_name
            all_preset_results.append(test_df)

    if not all_preset_results:
        print("Backtest could not be completed for any group.")
        return
        
    results_df = pd.concat(all_preset_results)
    results_df.to_csv('output/backtest_results_presets.csv', index=False)
    print("\n--- All preset backtests complete. Results saved. ---")


if __name__ == '__main__':
    run_backtest()
