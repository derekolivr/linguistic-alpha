# -----------------------------------------------------------------
# This is your main dashboard application file.
# To run it:
# 1. Open your terminal
# 2. Navigate to the root of your 'LINGUISTIC-ALPHA' project folder
# 3. Run the command: streamlit run dashboard/app.py
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# Add the parent directory to the Python path to allow importing 'analysis' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.feature_engineering import calculate_core_linguistic_features, calculate_catalyst_score, find_project_root, load_mock_data
from analysis.backtesting import get_stock_data

def run():
    st.set_page_config(
        page_title="Linguistic Alpha Signal Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Linguistic Alpha Signal Dashboard")
    st.markdown("""
    Welcome to the **Linguistic Alpha Signal Dashboard**. This platform analyzes the language used in corporate communications 
    to derive predictive signals about stock performance and volatility. Select a signal type and a company to explore the data.
    """)

    # --- Sidebar for User Input ---
    st.sidebar.header("Dashboard Controls")
    signal_type = st.sidebar.radio(
        "Select Signal Type",
        ('Core Signals (Company Filings)', 'Catalyst Signals (Crucial Events)'),
        help="**Core Signals** track the general communication style over time. **Catalyst Signals** analyze high-impact, event-driven texts."
    )

    # --- Load Data Based on Selection ---
    try:
        project_root = find_project_root()
        if 'Core' in signal_type:
            mock_filings = load_mock_data('mock_filings.json', project_root)
            features_df = calculate_core_linguistic_features(mock_filings)
            st.sidebar.info("Displaying **Core Linguistic Features** derived from standard company filings (e.g., 10-Ks).")
        else:
            mock_events = load_mock_data('mock_events.json', project_root)
            features_df = calculate_catalyst_score(mock_events)
            st.sidebar.info("Displaying **Catalyst Event Scores** derived from critical texts like short-seller reports.")
        
        features_df['date'] = pd.to_datetime(features_df['date'])
    except FileNotFoundError as e:
        st.error(f"Failed to load mock data. Please ensure the 'data' directory with mock JSON files exists. Error: {e}")
        return

    # --- Ticker Selection ---
    tickers = sorted(features_df['ticker'].unique())
    selected_ticker = st.sidebar.selectbox(
        'Select a Company Ticker',
        tickers,
        index=0,
        help="Choose a company to analyze from the mock dataset."
    )

    # Filter data for the selected ticker
    ticker_df = features_df[features_df['ticker'] == selected_ticker].copy()

    # --- Main Panel Display ---
    st.header(f"Analysis for: **{selected_ticker}**")

    if 'Core' in signal_type:
        display_core_signals(ticker_df, selected_ticker)
    else:
        display_catalyst_signals(ticker_df)

def display_core_signals(ticker_df, ticker):
    """Render the dashboard for Core Linguistic Signals."""
    st.subheader("Core Linguistic Features Over Time")
    
    # Melt the DataFrame to make it suitable for Altair
    df_melted = ticker_df.melt(
        id_vars=['date', 'ticker', 'speaker'], 
        value_vars=['complexity_score', 'sentiment_score', 'generalizing_score', 'self_reference_score'],
        var_name='Linguistic Feature', 
        value_name='Score'
    )

    # Interactive Chart
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Score:Q', title='Feature Score', scale=alt.Scale(zero=False)),
        color='Linguistic Feature:N',
        tooltip=['date:T', 'speaker:N', 'Linguistic Feature:N', 'Score:Q']
    ).interactive().properties(
        title=f"Linguistic Features for {ticker}"
    )
    st.altair_chart(chart, use_container_width=True)

    # Data Table
    st.subheader("Underlying Feature Data")
    st.dataframe(ticker_df)

def display_catalyst_signals(ticker_df):
    """Render the dashboard for Catalyst Event Signals."""
    st.subheader("Catalyst Event Scores")

    # Display scores in a clean format
    for _, row in ticker_df.iterrows():
        event_label = row.get('Attack_Score', row.get('Rebuttal_Severity_Score'))
        event_name = 'Attack Score' if 'Attack_Score' in row else 'Rebuttal Severity Score'
        
        with st.container():
            st.metric(label=f"**{row['event_type']} from {row['source']} on {row['date'].strftime('%Y-%m-%d')}**",
                      value=event_label,
                      help=f"A higher score indicates a more severe negative event. Specificity was {row['specificity_score']:.2%}.")
    
    # Data Table
    st.subheader("Underlying Event Data")
    st.dataframe(ticker_df)


if __name__ == "__main__":
    run()