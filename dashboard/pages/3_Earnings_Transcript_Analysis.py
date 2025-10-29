import streamlit as st
import pandas as pd
import plotly.express as px
import os
import statsmodels.api as sm

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

    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ("Raw Values", "Normalized (Z-score)")
    )

    # --- Single-Company Time Series Analysis ---
    st.header(f"Time Series Analysis for {selected_ticker}")
    company_df = df[df['ticker'] == selected_ticker].sort_values('date')
    
    # Let user select a feature to plot
    if analysis_mode == "Raw Values":
        feature_cols = [col for col in df.columns if not col.endswith('_zscore') and col not in ['ticker', 'date', 'speaker', 'next_quarter_return', 'next_quarter_volatility']]
    else: # Normalized
        feature_cols = [col for col in df.columns if col.endswith('_zscore')]

    feature_to_plot = st.selectbox(
        f"Select a Linguistic Feature to Analyze ({analysis_mode})",
        feature_cols
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
        
        if analysis_mode == "Raw Values":
            correlation_feature_cols = [col for col in df.columns if 'score' in col or 'ratio' in col or 'density' in col and not col.endswith('_zscore')]
        else: # Normalized
            correlation_feature_cols = [col for col in df.columns if col.endswith('_zscore')]

        correlation_feature = st.selectbox(
            f"Select Feature for Correlation with Next Quarter Return ({analysis_mode})",
            correlation_feature_cols
        )
        
        if correlation_feature:
            # --- R-squared Calculation ---
            plot_df = df.dropna(subset=[correlation_feature, 'next_quarter_return'])
            X = plot_df[correlation_feature]
            y = plot_df['next_quarter_return']
            
            # Add a constant for the intercept
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            
            st.metric(label="R-squared", value=f"{r_squared:.4f}")
            
            fig_scatter = px.scatter(
                plot_df,
                x=correlation_feature, 
                y='next_quarter_return',
                hover_data=['ticker', 'date'],
                title=f'Correlation: {correlation_feature} vs. Next Quarter Return',
                trendline="ols"  # Ordinary Least Squares trendline
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
