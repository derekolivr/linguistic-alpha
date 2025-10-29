import pandas as pd
import statsmodels.api as sm
import os

def evaluate_all_features():
    """
    Calculates the R-squared value for every feature against the next quarter return
    and prints a ranked list of the most predictive features.
    """
    file_path = 'output/transcript_features_with_performance.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please run the main pipeline first.")
        return

    df = pd.read_csv(file_path)

    # Identify all potential feature columns (linguistic metrics)
    # Exclude identifiers, target variables, and other non-feature columns
    excluded_cols = ['ticker', 'date', 'speaker', 'next_quarter_return', 'next_quarter_volatility']
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    results_return = []
    results_volatility = []

    for feature in feature_cols:
        # --- Analysis vs. Return ---
        df_return = df[[feature, 'next_quarter_return']].dropna()
        if len(df_return) >= 2:
            X = sm.add_constant(df_return[feature])
            y = df_return['next_quarter_return']
            model_return = sm.OLS(y, X).fit()
            results_return.append({'feature': feature, 'r_squared': model_return.rsquared})

        # --- Analysis vs. Volatility ---
        df_vol = df[[feature, 'next_quarter_volatility']].dropna()
        if len(df_vol) >= 2:
            X = sm.add_constant(df_vol[feature])
            y = df_vol['next_quarter_volatility']
            model_vol = sm.OLS(y, X).fit()
            results_volatility.append({'feature': feature, 'r_squared': model_vol.rsquared})

    # --- Print Ranked Results ---
    if results_return:
        ranked_return = sorted(results_return, key=lambda x: x['r_squared'], reverse=True)
        print("--- Feature Power vs. Next Quarter RETURN ---")
        for result in ranked_return:
            print(f"{result['feature']:<40} | R-squared: {result['r_squared']:.4f}")

    if results_volatility:
        ranked_volatility = sorted(results_volatility, key=lambda x: x['r_squared'], reverse=True)
        print("\n--- Feature Power vs. Next Quarter VOLATILITY ---")
        for result in ranked_volatility:
            print(f"{result['feature']:<40} | R-squared: {result['r_squared']:.4f}")


if __name__ == "__main__":
    evaluate_all_features()
