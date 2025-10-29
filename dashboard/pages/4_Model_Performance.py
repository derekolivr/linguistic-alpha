import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import plotly.figure_factory as ff
import numpy as np

st.set_page_config(layout="wide", page_title="Model Performance")

st.title("🤖 Model Performance Backtest")

@st.cache_data
def load_data_and_predictions():
    # Load the main data file
    data_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(data_path):
        st.error("Data file not found. Please run the main pipeline first.")
        return None, None, None

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the 2024 test data
    test_df = df[df['date'].dt.year == 2024].copy()
    if test_df.empty:
        st.warning("No 2024 data available for backtesting.")
        return None, None, None
        
    feature_cols = [col for col in df.columns if col.endswith('_zscore')]
    X_test = test_df[feature_cols]

    # Load models and make predictions
    predictions = {}
    
    # Return model
    return_model_path = 'output/return_classifier.joblib'
    if os.path.exists(return_model_path):
        return_model = joblib.load(return_model_path)
        predictions['return_pred'] = return_model.predict(X_test)
        predictions['return_proba'] = return_model.predict_proba(X_test)[:, 1]

    # Volatility model
    volatility_model_path = 'output/volatility_classifier.joblib'
    if os.path.exists(volatility_model_path):
        volatility_model = joblib.load(volatility_model_path)
        predictions['volatility_pred'] = volatility_model.predict(X_test)
        predictions['volatility_proba'] = volatility_model.predict_proba(X_test)[:, 1]
        
    # Add predictions to the dataframe
    test_df = test_df.reset_index(drop=True)
    pred_df = pd.DataFrame(predictions)
    results_df = pd.concat([test_df, pred_df], axis=1)
    
    return results_df, return_model, volatility_model

results_df, return_model, volatility_model = load_data_and_predictions()

if results_df is not None:
    
    st.header("Return Direction Prediction")
    if 'return_pred' in results_df.columns:
        y_true = results_df['return_class']
        y_pred = results_df['return_pred']
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        col2.metric("AUC Score", f"{roc_auc_score(y_true, results_df['return_proba']):.4f}")
        
        # Display Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = ff.create_annotated_heatmap(z=cm, x=['Predicted Down', 'Predicted Up'], y=['Actual Down', 'Actual Up'], colorscale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Prediction Details")
        st.dataframe(results_df[['ticker', 'date', 'return_class', 'return_pred', 'return_proba']])

    st.header("Volatility Regime Prediction")
    if 'volatility_pred' in results_df.columns:
        y_true = results_df['volatility_class']
        y_pred = results_df['volatility_pred']
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        col2.metric("AUC Score", f"{roc_auc_score(y_true, results_df['volatility_proba']):.4f}")

        # Display Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = ff.create_annotated_heatmap(z=cm, x=['Predicted Low Vol', 'Predicted High Vol'], y=['Actual Low Vol', 'Actual High Vol'], colorscale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction Details")
        st.dataframe(results_df[['ticker', 'date', 'volatility_class', 'volatility_pred', 'volatility_proba']])
