import streamlit as st
from dashboard.helpers import predict_stock_price, load_data, setup_path

st.set_page_config(
    page_title="Company Analysis",
    page_icon="🏢",
)

project_root = setup_path()
df = load_data(project_root)

def render_company_selection_page():
    """Renders the page to select a company from the chosen index."""
    selected_index = st.session_state.get('selected_index', 'S&P 500') # Default for direct access
    st.title(f"{selected_index} - Top Companies")
    st.info(f"Top 10 performing companies in the {selected_index}, based on market cap.")

    index_tickers = {
        'S&P 500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'V'],
        'NASDAQ 100': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'PEP', 'COST']
    }
    
    top_10 = index_tickers[selected_index]

    st.subheader("Select a Company for a Detailed Analysis")
    selected_ticker = st.selectbox("Choose a company from the list:", top_10)
    
    if st.button(f"Analyze {selected_ticker}"):
        st.session_state['selected_ticker'] = selected_ticker
        st.rerun()

def render_detail_page(ticker):
    """The deep-dive page for a single company."""
    st.title(f"🔍 Detailed Analysis: {ticker}")
    
    if st.button("← Back to Company Selection"):
        del st.session_state['selected_ticker']
        st.rerun()
    
    if df is not None:
        ticker_df = df[df['ticker'] == ticker].sort_values(by='date', ascending=False)
        if not ticker_df.empty:
            latest_filing = ticker_df.iloc[0]
            st.header(f"Key Metrics (Latest Filing: {latest_filing['date'].strftime('%Y-%m-%d')})")
            c1, c2, c3 = st.columns(3)
            c1.metric("MD&A Sentiment", f"{latest_filing['sentiment_score_mda']:.2f}")
            c2.metric("Complexity Delta", f"{latest_filing['complexity_delta']:.2f}")
            c3.metric("Risk Text Change", f"{latest_filing['risk_text_change_score_risk']:.2f}")
        else:
            st.warning(f"No linguistic data found for {ticker}.")
    else:
        st.warning("Linguistic data file not found.")

    st.header("📈 Stock Price Prediction")
    if st.button("Predict Next 6 Months"):
        with st.spinner("Generating forecast..."):
            forecast_fig = predict_stock_price(ticker)
            st.plotly_chart(forecast_fig)

# --- Main Page Logic ---
if not st.session_state.get('authenticated'):
    st.error("You must be logged in to access this page.")
    st.page_link("app.py", label="Go to Login", icon="🔓")
elif 'selected_ticker' in st.session_state:
    render_detail_page(st.session_state['selected_ticker'])
else:
    render_company_selection_page()
