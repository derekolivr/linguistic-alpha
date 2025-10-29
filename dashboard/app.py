import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# --- Robust Path Setup ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    project_root = os.getcwd()
    if 'dashboard' in project_root:
        project_root = project_root.split('dashboard')[0]

from utils.calculations import calculate_historical_pnl

# --- 1. DATA LOADING & CACHING (Moved inside the dashboard function) ---
@st.cache_data
def load_data(root_path):
    """Loads and caches the final feature data from the output CSV."""
    csv_path = os.path.join(root_path, 'output', 'final_linguistic_features.csv')
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df['best_score'] = (df['sentiment_score_mda'] - df['complexity_delta'] - df['risk_text_change_score_risk']).fillna(0)
        return df
    except FileNotFoundError:
        return None

# --- 2. UI PAGES ---

def render_landing_page():
    """Renders a visually appealing landing page BEFORE login."""
    st.set_page_config(layout="wide", page_title="Welcome to Linguistic Alpha")
    
    # Use columns for a centered, clean look
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("📈 Linguistic Alpha")
        st.subheader("Go Beyond the Numbers. Analyze the Narrative.")
        
        st.write(
            """
            Welcome to Linguistic Alpha, a quantitative analysis platform that leverages Natural Language Processing (NLP) 
            to decode the language of corporate finance. We analyze SEC filings to generate unique insights into company 
            performance, risk, and strategic direction.
            """
        )
        
        st.markdown("---")
        
        st.info(
            """
            **Our Approach:**
            - **Sentiment Analysis:** We measure the tone of management's discussion.
            - **Complexity Scoring:** We identify when language becomes unusually complex or obfuscated.
            - **Risk Disclosure Tracking:** We monitor changes in disclosed risks quarter over quarter.
            """
        )
        
        # The button that will trigger the login state
        if st.button("Login to Access the Dashboard", key="login_button", type="primary"):
            st.session_state['app_state'] = 'login'
            st.experimental_rerun()

def render_login_page():
    """Renders the login screen."""
    st.set_page_config(layout="centered", page_title="Login")
    
    st.title("User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # Placeholder for secure authentication
            if username == "admin" and password == "lingalpha":
                st.session_state['authenticated'] = True
                st.session_state['app_state'] = 'dashboard'
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password")

def render_dashboard(df):
    """Renders the main dashboard application AFTER login."""
    st.set_page_config(layout="wide", page_title="Linguistic Alpha Dashboard")

    st.sidebar.title(f"Welcome, User!")
    st.sidebar.markdown("---")
    
    # --- Page Router for the main app (Index vs Detail) ---
    if 'page' not in st.session_state:
        st.session_state['page'] = 'index'

    if st.session_state['page'] == 'index':
        render_index_page(df)
    elif st.session_state['page'] == 'detail':
        render_detail_page(df, st.session_state['selected_ticker'])

def render_index_page(df):
    """The main index/ranking page of the dashboard."""
    st.title("Linguistic Alpha 10 Index")
    st.info("Ranking of companies by the linguistic quality of their latest SEC filings.")

    latest_filings = df.loc[df.groupby('ticker')['date'].idxmax()].copy()
    top_10 = latest_filings.sort_values(by='best_score', ascending=False).head(10)

    st.subheader("Top 10 Companies by Linguistic Score")
    
    display_cols = {'ticker': 'Ticker', 'best_score': 'Linguistic Score', 'sentiment_score_mda': 'MD&A Sentiment'}
    st.dataframe(top_10[display_cols.keys()].rename(columns=display_cols).set_index('Ticker'))

    st.subheader("Select a Company for a Detailed Analysis")
    selected_ticker = st.selectbox("Choose a company from the list:", top_10['ticker'].tolist())
    
    if st.button(f"Analyze {selected_ticker}"):
        st.session_state['page'] = 'detail'
        st.session_state['selected_ticker'] = selected_ticker
        st.experimental_rerun()

def render_detail_page(df, ticker):
    """The deep-dive page for a single company."""
    st.title(f"🔍 Detailed Analysis: {ticker}")
    
    if st.button("← Back to Index"):
        st.session_state['page'] = 'index'
        st.experimental_rerun()
    
    ticker_df = df[df['ticker'] == ticker].sort_values(by='date', ascending=False)
    latest_filing = ticker_df.iloc[0]

    st.header(f"Key Metrics (Latest Filing: {latest_filing['date'].strftime('%Y-%m-%d')})")
    c1, c2, c3 = st.columns(3)
    c1.metric("MD&A Sentiment", f"{latest_filing['sentiment_score_mda']:.2f}")
    c2.metric("Complexity Delta", f"{latest_filing['complexity_delta']:.2f}")
    c3.metric("Risk Text Change", f"{latest_filing['risk_text_change_score_risk']:.2f}")

    st.subheader("Simulate Past Performance")
    investment_amount = st.number_input("If you had invested this amount:", value=10000)
    if st.button("Calculate 6-Month Historical P&L"):
        with st.spinner("Calculating..."):
            pnl_data = calculate_historical_pnl(ticker, investment_amount)
            # ... (rest of P&L display logic) ...
            if "error" in pnl_data: st.error(pnl_data["error"])
            else:
                st.metric(label=f"Current Value", value=f"${pnl_data['current_value']:,.2f}", delta=f"${pnl_data['profit_loss']:,.2f}")
                st.line_chart(pnl_data['price_history'])


# --- 3. MAIN APP CONTROLLER ---
def main():
    """Controls the overall application state and routing."""
    # Initialize session state variables
    if 'app_state' not in st.session_state:
        st.session_state['app_state'] = 'landing'
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    # State-based routing
    if st.session_state['app_state'] == 'landing':
        render_landing_page()
    
    elif st.session_state['app_state'] == 'login':
        render_login_page()
        
    elif st.session_state['app_state'] == 'dashboard' and st.session_state['authenticated']:
        # Load the data ONLY after the user is authenticated
        df = load_data(project_root)
        if df is None or df.empty:
            st.error("FATAL: 'output/final_linguistic_features.csv' not found. Please run the main pipeline (`run_pipeline.py`).")
            st.stop()
        render_dashboard(df)
        
    else: # Default to landing if state is invalid
        st.session_state['app_state'] = 'landing'
        st.experimental_rerun()


if __name__ == "__main__":
    main()