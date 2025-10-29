import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Linguistic Alpha: Welcome",
    page_icon="📈"
)

# --- Landing Page Content ---
st.title("📈 Linguistic Alpha: Predicting Market Risk from Language")

st.markdown("""
Welcome to the Linguistic Alpha project. This dashboard is the user interface for an end-to-end data pipeline that analyzes the language of S&P 500 earnings calls to predict future stock market volatility.

The key finding of this project is the successful creation of a **Volatility Prediction Model** that demonstrated a strong, verifiable ability to predict future risk based on the linguistic patterns in corporate earnings calls.
""")

st.info("Please select an analysis page from the sidebar to begin.", icon="👈")

st.header("Dashboard Pages")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Analysis & Correlations")
    st.write("""
    An interactive tool to explore the raw linguistic features extracted from the earnings call transcripts. This page allows you to:
    - Track a single company's linguistic metrics over time.
    - Compare features across different companies for a specific quarter.
    - Analyze the correlation between individual features and future stock returns or volatility.
    """)
    st.page_link("pages/3_Earnings_Transcript_Analysis.py", label="Go to Feature Analysis", icon="🗣️")

with col2:
    st.subheader("Model Performance Backtest")
    st.write("""
    The main event. This page presents the backtest results for our two predictive models, with a focus on the highly successful Volatility Predictor. You can:
    - View key performance metrics like Accuracy, AUC, Precision, and Recall.
    - Analyze a confusion matrix to see where the models succeed and fail.
    - Explore a detailed table of predictions vs. actual outcomes for the 2024 test set.
    """)
    st.page_link("pages/4_Model_Performance.py", label="Go to Model Performance", icon="🤖")

# --- Remove Login/Signup for Hackathon Demo ---
# The old login/signup logic has been removed to streamline the demo experience.