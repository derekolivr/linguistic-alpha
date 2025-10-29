import streamlit as st
import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide", page_title="Model Backtest Scenarios")

st.title("🤖 Model Backtest Scenarios")
st.markdown("Explore the model's performance on different pre-computed scenarios. Each scenario trains and tests the models on a specific group of companies.")

@st.cache_data
def load_all_results():
    data_path = 'output/backtest_results_presets.csv'
    if not os.path.exists(data_path):
        st.error("Preset backtest results file not found. Please run `run_pipeline.py` first.")
        return None
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

all_results_df = load_all_results()

if all_results_df is not None:
    
    # --- UI Controls ---
    preset_groups = sorted(all_results_df['preset_group'].unique())
    selected_group = st.selectbox(
        "Select a Scenario to View:",
        preset_groups
    )
    
    # Filter results for the selected group
    results_df = all_results_df[all_results_df['preset_group'] == selected_group]
    
    st.header(f"Showing Results for: {selected_group}")
    
    if not results_df.empty:
        # The rest of the display logic is the same as before, using the filtered results_df
        vol_tab, return_tab = st.tabs(["Volatility Model Performance", "Return Model Performance"])

        with vol_tab:
            # Display metrics, confusion matrix, and timeline for volatility...
            st.header("Volatility Regime Prediction")
            y_true_vol = results_df['volatility_class']
            y_pred_vol = results_df['volatility_pred']
            
            # Display Metrics
            st.subheader("Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_true_vol, y_pred_vol):.2%}", help="Overall percentage of correct predictions.")
            col2.metric("AUC Score", f"{roc_auc_score(y_true_vol, results_df['volatility_proba']):.4f}", help="Ability to distinguish between classes. > 0.7 is good.")
            col3.metric("Precision", f"{precision_score(y_true_vol, y_pred_vol, zero_division=0):.2%}", help="Of all 'High Volatility' predictions, how many were correct?")
            col4.metric("Recall", f"{recall_score(y_true_vol, y_pred_vol, zero_division=0):.2%}", help="Of all actual 'High Volatility' events, how many did the model catch?")

            st.subheader("Confusion Matrix")
            cm_vol = confusion_matrix(y_true_vol, y_pred_vol)
            fig_cm_vol = ff.create_annotated_heatmap(z=cm_vol, x=['Predicted Low Vol', 'Predicted High Vol'], y=['Actual Low Vol', 'Actual High Vol'], colorscale='Oranges')
            st.plotly_chart(fig_cm_vol, use_container_width=True)

            st.subheader("Risk Signal Timeline")
            # This logic remains the same as previously implemented
            tickers_in_results = sorted(results_df['ticker'].unique())
            selected_ticker_timeline = st.selectbox("Select a Ticker to Visualize", tickers_in_results, key='vol_timeline_ticker')
            
            company_df = results_df[results_df['ticker'] == selected_ticker_timeline].sort_values('date')

            if not company_df.empty:
                fig_timeline = go.Figure()

                fig_timeline.add_trace(go.Scatter(
                    x=company_df['date'], y=company_df['composite_risk_score_zscore'],
                    mode='lines+markers', name='Composite Risk Score (Z-Score)',
                    line=dict(color='royalblue')
                ))
                actual_high_vol = company_df[company_df['volatility_class'] == 1]
                fig_timeline.add_trace(go.Scatter(
                    x=actual_high_vol['date'], y=actual_high_vol['composite_risk_score_zscore'],
                    mode='markers', name='Actual High Volatility',
                    marker=dict(color='red', size=12, symbol='x')
                ))
                predicted_high_vol = company_df[company_df['volatility_pred'] == 1]
                fig_timeline.add_trace(go.Scatter(
                    x=predicted_high_vol['date'], y=predicted_high_vol['composite_risk_score_zscore'],
                    mode='markers', name='Predicted High Volatility',
                    marker=dict(color='lightgreen', size=8, symbol='circle', line=dict(width=1, color='black'))
                ))

                fig_timeline.update_layout(title=f"Risk Signal Timeline for {selected_ticker_timeline}",
                                         xaxis_title="Date",
                                         yaxis_title="Composite Risk Score (Deviation from Norm)")
                st.plotly_chart(fig_timeline, use_container_width=True)

        with return_tab:
            # Display metrics and confusion matrix for returns...
            st.header("Return Direction Prediction")
            y_true_return = results_df['return_class']
            y_pred_return = results_df['return_pred']

            st.subheader("Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_true_return, y_pred_return):.2%}")
            col2.metric("AUC Score", f"{roc_auc_score(y_true_return, results_df['return_proba']):.4f}")
            col3.metric("Precision", f"{precision_score(y_true_return, y_pred_return, zero_division=0):.2%}")
            col4.metric("Recall", f"{recall_score(y_true_return, y_pred_return, zero_division=0):.2%}")

            st.subheader("Confusion Matrix")
            cm_return = confusion_matrix(y_true_return, y_pred_return)
            fig_cm_return = ff.create_annotated_heatmap(z=cm_return, x=['Predicted Down', 'Predicted Up'], y=['Actual Down', 'Actual Up'], colorscale='Blues')
            st.plotly_chart(fig_cm_return, use_container_width=True)

    else:
        st.warning(f"No results available for the '{selected_group}' scenario.")

else:
    st.error("Could not load backtest results. Please ensure the pipeline has been run successfully.")
