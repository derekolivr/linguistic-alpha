import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

def train_all_models():
    """
    Trains the final "production" models on all available historical data
    and saves them for future prediction.
    """
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    df = pd.read_csv(file_path).dropna(subset=['return_class', 'volatility_class'])

    # --- Feature Selection ---
    feature_cols = [col for col in df.columns if col.endswith('_zscore')]
    X_full = df[feature_cols].fillna(0)

    # --- Train and Save Return Production Model ---
    print("Training FINAL return prediction model on all data...")
    y_return = df['return_class']
    scale_pos_weight_return = (y_return == 0).sum() / (y_return == 1).sum()
    estimators_return = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_return))
    ]
    stacking_classifier_return = StackingClassifier(estimators=estimators_return, final_estimator=LogisticRegression())
    return_model = stacking_classifier_return.fit(X_full, y_return)
    joblib.dump(return_model, 'output/return_classifier_production.joblib')
    print("Return production model trained and saved.")

    # --- Train and Save Volatility Production Model ---
    print("\nTraining FINAL volatility prediction model on all data...")
    y_vol = df['volatility_class']
    scale_pos_weight_vol = (y_vol == 0).sum() / (y_vol == 1).sum()
    estimators_vol = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_vol))
    ]
    stacking_classifier_vol = StackingClassifier(estimators=estimators_vol, final_estimator=LogisticRegression())
    volatility_model = stacking_classifier_vol.fit(X_full, y_vol)
    joblib.dump(volatility_model, 'output/volatility_classifier_production.joblib')
    print("Volatility production model trained and saved.")

if __name__ == '__main__':
    train_all_models()
