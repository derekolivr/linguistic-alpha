import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="Earnings Transcript Analysis")

st.title("🗣️ Earnings Transcript Analysis")

@st.cache_data
def load_data():
    """Loads the processed transcript features and performance data."""
    file_path = 'output/transcript_features_with_performance.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error("Data file not found. Please run the feature engineering pipeline first.")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Filters")
    tickers = df['ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select a Company", tickers)

    # --- Single-Company Time Series Analysis ---
    st.header(f"Time Series Analysis for {selected_ticker}")
    company_df = df[df['ticker'] == selected_ticker].sort_values('date')
    
    # Let user select a feature to plot
    feature_to_plot = st.selectbox(
        "Select a Linguistic Feature to Analyze",
        [col for col in df.columns if col not in ['ticker', 'date', 'speaker']]
    )

    fig = px.line(company_df, x='date', y=feature_to_plot, title=f'{feature_to_plot} Over Time for {selected_ticker}', markers=True)
    fig.update_layout(xaxis_title="Date", yaxis_title=feature_to_plot)
    st.plotly_chart(fig, use_container_width=True)

    # --- Cross-Sectional and Correlation Analysis ---
    st.header("Comparative and Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cross-sectional comparison for a specific quarter
        st.subheader("Cross-Sectional Company Comparison")
        quarters = pd.to_datetime(df['date']).dt.to_period('Q').unique().astype(str)
        selected_quarter = st.selectbox("Select a Quarter", quarters)
        
        quarter_df = df[pd.to_datetime(df['date']).dt.to_period('Q').astype(str) == selected_quarter]
        
        if not quarter_df.empty:
            fig_bar = px.bar(quarter_df, x='ticker', y=feature_to_plot, title=f'{feature_to_plot} Across Companies for {selected_quarter}')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No data available for the selected quarter.")

    with col2:
        # Correlation plot
        st.subheader("Correlation Analysis")
        correlation_feature = st.selectbox(
            "Select Feature for Correlation with Next Quarter Return",
            [col for col in df.columns if 'score' in col or 'ratio' in col or 'density' in col]
        )
        
        if correlation_feature:
            fig_scatter = px.scatter(
                df.dropna(subset=[correlation_feature, 'next_quarter_return']),
                x=correlation_feature, 
                y='next_quarter_return',
                hover_data=['ticker', 'date'],
                title=f'Correlation: {correlation_feature} vs. Next Quarter Return',
                trendline="ols"  # Ordinary Least Squares trendline
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
