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
    
    # --- Feature Selection ---
    feature_cols = [col for col in df.columns if col.endswith('_zscore')]
    
    # --- Train and Save Return Prediction Model ---
    print("Training return prediction model...")
    X_train_return = train_df[feature_cols].fillna(0)
    y_train_return = train_df['return_class']
    
    if not y_train_return.empty and (y_train_return == 1).sum() > 0:
        scale_pos_weight_return = (y_train_return == 0).sum() / (y_train_return == 1).sum()
        estimators_return = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_return))
        ]
        stacking_classifier_return = StackingClassifier(estimators=estimators_return, final_estimator=LogisticRegression())
        return_model = stacking_classifier_return.fit(X_train_return, y_train_return)
        joblib.dump(return_model, 'output/return_classifier.joblib')
        print("Return model trained and saved successfully.")
    else:
        print("Could not train return model due to lack of data or positive samples.")

    # --- Train and Save Volatility Prediction Model ---
    print("\nTraining volatility prediction model...")
    X_train_vol = train_df[feature_cols].fillna(0)
    y_train_vol = train_df['volatility_class']

    if not y_train_vol.empty and (y_train_vol == 1).sum() > 0:
        scale_pos_weight_vol = (y_train_vol == 0).sum() / (y_train_vol == 1).sum()
        estimators_vol = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_vol))
        ]
        stacking_classifier_vol = StackingClassifier(estimators=estimators_vol, final_estimator=LogisticRegression())
        volatility_model = stacking_classifier_vol.fit(X_train_vol, y_train_vol)
        joblib.dump(volatility_model, 'output/volatility_classifier.joblib')
        print("Volatility model trained and saved successfully.")
    else:
        print("Could not train volatility model due to lack of data or positive samples.")

if __name__ == '__main__':
    train_all_models()
