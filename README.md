# Linguistic Alpha: Earnings Call Analysis Engine

This project leverages Natural Language Processing (NLP) and Machine Learning to analyze the language of S&P 500 earnings call transcripts and predict future stock volatility. It features an end-to-end pipeline that automates data collection, feature engineering, model training, and performance evaluation.

The key finding of this project is the successful creation of a **Volatility Prediction Model** that demonstrated a strong, verifiable ability to predict future risk based on the linguistic patterns in corporate earnings calls.

## Features

- **Automated Data Pipeline**: Downloads and processes years of earnings call transcripts from the `kurry/sp500_earnings_transcripts` dataset on Hugging Face.
- **Advanced Feature Engineering**: Calculates dozens of linguistic features, including sentiment, complexity, and risk keyword density, and normalizes them using Z-scores to measure deviation from a company's historical average.
- **Dual Predictive Models**: Trains two separate ensemble machine learning models:
  - **Volatility Prediction Model (Successful)**: Predicts whether the next quarter will be a high or low volatility period with high accuracy.
  - **Return Direction Model (Experimental)**: Predicts whether the next quarter's stock return will be positive or negative.
- **Rigorous Backtesting**: Utilizes a strict temporal hold-out set (2024 data) to evaluate model performance on completely unseen data.
- **Interactive Dashboard**: A multi-page Streamlit application to visualize the linguistic features and the final model performance metrics.

## Project Structure

```
linguistic-alpha/
├── analysis/
│   ├── data_loader.py         # Downloads and processes transcript data
│   ├── transcript_feature_engineering.py # Calculates linguistic features
│   ├── model_training.py      # Trains and saves the ML models
│   └── backtest.py            # Evaluates models on hold-out data
├── dashboard/
│   ├── app.py                 # Main Streamlit app
│   └── pages/                 # Dashboard pages for analysis and model performance
├── output/                    # Stores generated data and trained models
├── run_pipeline.py            # Main script to run the entire pipeline
├── requirements.txt
└── README.md
```

## How to Run

1.  **Clone the Repository**:

    ```bash
    git clone <your-repo-url>
    cd linguistic-alpha
    ```

2.  **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Full Pipeline**:
    This command will download the data, perform feature engineering, train the models, and run the backtest. This will take some time as it processes over a decade of data.

    ```bash
    python run_pipeline.py
    ```

5.  **Launch the Dashboard**:
    After the pipeline completes, you can view the results.
    ```bash
    streamlit run dashboard/app.py
    ```

## Key Findings

The primary success of this project is the **Volatility Prediction Model**, which achieved an **Accuracy of 81.25%** and an **AUC Score of 0.85** on the 2024 hold-out set. This demonstrates a strong predictive signal in linguistic data for forecasting future market risk. The Return Prediction Model did not show significant predictive power, confirming the difficulty of predicting stock direction.
