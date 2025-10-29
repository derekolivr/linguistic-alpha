# Linguistic Alpha: Stock Market Dashboard

Welcome to the Linguistic Alpha Stock Market Dashboard! This application, built for a hackathon, leverages Natural Language Processing (NLP) to derive insights from financial documents and combines them with market data. It allows users to create profiles, analyze top companies from major stock indices, and predict future stock performance.

## Features

- **User Authentication**: Secure sign-up and login functionality.
- **Multi-Page App**: A clean, organized user interface with separate pages for different features.
- **Stock Index Selection**: Analyze top companies from the S&P 500 and NASDAQ 100.
- **Company Analysis**: View detailed linguistic metrics for selected companies (data permitting).
- **AI-Powered Stock Prediction**: Generate a 6-month stock price forecast using a time-series model (Prophet).

## Project Structure

```
linguistic-alpha/
├── dashboard/
│   ├── app.py              # Main app: handles login/signup
│   ├── helpers.py          # Shared functions (data loading, prediction)
│   └── pages/
│       ├── 1_Index_Selection.py
│       └── 2_Company_Analysis.py
├── data/
├── output/
├── requirements.txt
└── README.md
```

## Installation

1.  **Clone the Repository**:

    ```bash
    git clone <your-repo-url>
    cd linguistic-alpha
    ```

2.  **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Application**:

    ```bash
    streamlit run dashboard/app.py
    ```

2.  **Access the Dashboard**:
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment on Streamlit Community Cloud

1.  **Push to GitHub**: Make sure your code is in a public GitHub repository.

2.  **Deploy on Streamlit**:
    - Sign in to [share.streamlit.io](https://share.streamlit.io/).
    - Click on "New app" and select your repository and branch.
    - Set the **Main file path** to `dashboard/app.py`.
    - Click "Deploy!".
