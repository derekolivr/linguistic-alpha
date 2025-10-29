import streamlit as st
import pandas as pd
from helpers import setup_path

# --- User Management (for demo purposes) ---
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': 'lingalpha'}

project_root = setup_path()

# --- UI PAGES ---

def render_landing_page():
    """Renders a visually appealing landing page BEFORE login."""
    st.set_page_config(layout="wide", page_title="Welcome to Linguistic Alpha")
    
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
        
        c1, c2 = st.columns(2)
        if c1.button("Login", key="login_button", type="primary", use_container_width=True):
            st.session_state['app_state'] = 'login'
            st.rerun()
        
        if c2.button("Sign Up", key="signup_button", use_container_width=True):
            st.session_state['app_state'] = 'signup'
            st.rerun()

def render_login_page():
    """Renders the login screen."""
    st.set_page_config(layout="centered", page_title="Login")
    
    st.title("User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in st.session_state['users'] and st.session_state['users'][username] == password:
                st.session_state['authenticated'] = True
                st.session_state['app_state'] = 'dashboard'
                st.rerun()
            else:
                st.error("Incorrect username or password")
    
    if st.button("← Back to Home"):
        st.session_state['app_state'] = 'landing'
        st.rerun()

def render_signup_page():
    """Renders the signup screen."""
    st.set_page_config(layout="centered", page_title="Sign Up")
    
    st.title("Create a New Account")
    
    with st.form("signup_form"):
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_submitted = st.form_submit_button("Sign Up")
        
        if signup_submitted:
            if new_username in st.session_state['users']:
                st.error("Username already exists. Please choose another one.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty.")
            else:
                st.session_state['users'][new_username] = new_password
                st.success("Account created successfully! Please login.")
                st.session_state['app_state'] = 'login'
                st.rerun()

    if st.button("← Back to Home"):
        st.session_state['app_state'] = 'landing'
        st.rerun()

def render_dashboard():
    """Renders the main dashboard landing area after login."""
    st.set_page_config(layout="wide", page_title="Linguistic Alpha Dashboard")
    st.title("Welcome to the Dashboard")
    st.info("Please select a page from the sidebar to begin your analysis.")
    st.page_link("pages/1_Index_Selection.py", label="Start Analysis", icon="📈")
    st.page_link("pages/3_Earnings_Transcript_Analysis.py", label="Earnings Transcript Analysis", icon="🗣️")

    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['app_state'] = 'landing'
        st.rerun()


# --- MAIN APP CONTROLLER ---
def main():
    """Controls the overall application state and routing."""
    if 'app_state' not in st.session_state:
        st.session_state['app_state'] = 'landing'
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if st.session_state['authenticated']:
        render_dashboard()
    else:
        if st.session_state['app_state'] == 'landing':
            render_landing_page()
        elif st.session_state['app_state'] == 'login':
            render_login_page()
        elif st.session_state['app_state'] == 'signup':
            render_signup_page()
        else:
            render_landing_page()


if __name__ == "__main__":
    main()