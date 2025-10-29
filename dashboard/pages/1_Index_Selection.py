import streamlit as st

st.set_page_config(
    page_title="Index Selection",
    page_icon="📈",
)

def render_index_selection_page():
    """Renders the page for selecting a stock market index."""
    st.title("📈 Stock Market Index Analysis")
    st.info("Select a stock market index to view the top 10 performing companies.")

    index_options = ['S&P 500', 'NASDAQ 100']
    selected_index = st.selectbox("Choose an index:", index_options)

    if st.button(f"Analyze {selected_index}"):
        st.session_state['selected_index'] = selected_index
        st.switch_page("pages/2_Company_Analysis.py")

if st.session_state.get('authenticated'):
    render_index_selection_page()
else:
    st.error("You must be logged in to access this page.")
    st.page_link("app.py", label="Go to Login", icon="🔓")
