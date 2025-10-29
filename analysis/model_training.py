import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

def train_all_models():
    """
    Trains and saves two separate stacking ensemble models: one for predicting
    return direction and one for predicting volatility regime.
    """
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # --- Temporal Split ---
    train_df = df[df['date'].dt.year < 2024]
    # Note: The test set is implicitly the 2024 data, which will be used in backtest.py

    # --- Feature Selection ---
    # We use z-scores and the composite score for prediction
    feature_cols = [col for col in df.columns if col.endswith('_zscore')]
    
    # --- Define Models ---
    # Base estimators for the stacking model
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    
    # Stacking classifier definition
    stacking_classifier = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

    # --- Train and Save Return Prediction Model ---
    print("Training return prediction model...")
    X_train_return = train_df[feature_cols]
    y_train_return = train_df['return_class']
    
    # Ensure we have data to train on
    if not X_train_return.empty and not y_train_return.empty:
        return_model = stacking_classifier.fit(X_train_return, y_train_return)
        joblib.dump(return_model, 'output/return_classifier.joblib')
        print("Return model trained and saved successfully.")
    else:
        print("Could not train return model due to lack of data.")

    # --- Train and Save Volatility Prediction Model ---
    print("\nTraining volatility prediction model...")
    X_train_vol = train_df[feature_cols]
    y_train_vol = train_df['volatility_class']
    
    # Ensure we have data to train on
    if not X_train_vol.empty and not y_train_vol.empty:
        volatility_model = stacking_classifier.fit(X_train_vol, y_train_vol)
        joblib.dump(volatility_model, 'output/volatility_classifier.joblib')
        print("Volatility model trained and saved successfully.")
    else:
        print("Could not train volatility model due to lack of data.")

if __name__ == '__main__':
    train_all_models()
