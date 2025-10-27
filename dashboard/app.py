# -----------------------------------------------------------------
# This is your main dashboard application file.
# To run it:
# 1. Open your terminal
# 2. Navigate to the root of your 'LINGUISTIC-ALPHA' project folder
# 3. Run the command: streamlit run dashboard/app.py
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import os
import sys

# --- Path Setup ---
# This is crucial for allowing this script to import modules from other folders
# like 'analysis' and 'utils'.
try:
    # This assumes the script is in 'dashboard/', so we go up one level to the root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    # Fallback for interactive environments
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)


# --- Import Your Custom Analysis Functions ---
from analysis.feature_engineering import (
    load_mock_data,
    calculate_core_linguistic_features,
    calculate_catalyst_score
)

# --- Import the Helper Function from the 'utils' package ---
# This function is no longer defined in this file. It is now a reusable utility.
from utils.calculations import calculate_historical_pnl


# ---------------- STREAMLIT APP MAIN BODY ----------------

st.set_page_config(layout="wide")
st.title("Linguistic Alpha: Analysis Dashboard")

st.info(
    "This dashboard analyzes company communications to identify linguistic risk factors. "
    "Select a company from the analysis table to simulate a historical investment."
)

# --- STEP 1: Run the Linguistic Analysis (using mock data for this example) ---
try:
    st.header("1. Linguistic Analysis Results")
    
    # Load the mock data for company filings
    # In a real application, this would come from your live parser.
    core_mock_data = load_mock_data('mock_filings.json', project_root)
    core_features_df = calculate_core_linguistic_features(core_mock_data)
    
    # Display the analysis table
    st.write("Below are the calculated linguistic risk scores for a sample of companies.")
    st.dataframe(core_features_df)
    
    # Create the dropdown for the user to select a ticker
    company_tickers = core_features_df['ticker'].unique().tolist()
    selected_ticker = st.selectbox(
        "Select a company to simulate a historical investment:",
        options=company_tickers
    )

    # --- STEP 2: The P&L Calculator Section ---
    # This entire block will only appear if a ticker has been selected.
    if selected_ticker:
        st.header(f"2. Historical P&L Simulation for {selected_ticker}")

        investment_amount = st.number_input(
            "If you had invested this amount 6 months ago:",
            min_value=100,
            value=10000,
            step=100,
            key=f"pnl_input_{selected_ticker}" # Add a key to prevent widget state issues
        )

        if st.button("Calculate Historical P&L"):
            with st.spinner(f"Fetching data and calculating P&L for {selected_ticker}..."):
                # Call the imported utility function
                pnl_data = calculate_historical_pnl(selected_ticker, investment_amount)

                if "error" in pnl_data:
                    st.error(pnl_data["error"])
                else:
                    # Display the results using st.metric for a nice visual
                    st.metric(
                        label="Current Value of Investment",
                        value=f"${pnl_data['current_value']:,.2f}",
                        delta=f"${pnl_data['profit_loss']:,.2f} ({pnl_data['percent_return']:.2%})"
                    )
                    
                    st.write(
                        f"An investment of **${pnl_data['initial_investment']:,.2f}** made on "
                        f"**{pnl_data['start_date']}** (at a price of ${pnl_data['initial_price']:.2f}) "
                        f"would be worth **${pnl_data['current_value']:,.2f}** today."
                    )

                    # Display a price chart for context
                    st.line_chart(pnl_data['price_history'])

# --- Error Handling ---
except FileNotFoundError:
    st.error(
        "Error: Could not find mock data files in the 'data/' directory. "
        "Please ensure 'data/mock_filings.json' exists in your project."
    )
except Exception as e:
    st.error(f"An unexpected error occurred. Please check the console for details. Error: {e}")