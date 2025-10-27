# utils/calculations.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def calculate_historical_pnl(ticker, investment_amount, lookback_months=6):
    """Calculates the P&L for a stock based on a historical investment."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_months * 30)

    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if stock_data.empty:
        return {"error": f"Could not download data for {ticker}."}

    initial_price = stock_data['Close'].iloc[0]
    current_price = stock_data['Close'].iloc[-1]

    num_shares = investment_amount / initial_price
    current_value = num_shares * current_price
    profit_loss = current_value - investment_amount
    percent_return = (profit_loss / investment_amount)

    return {
        "ticker": ticker,
        "start_date": stock_data.index[0].strftime('%Y-%m-%d'),
        "end_date": stock_data.index[-1].strftime('%Y-%m-%d'),
        "initial_investment": investment_amount,
        "initial_price": initial_price,
        "current_value": current_value,
        "current_price": current_price,
        "profit_loss": profit_loss,
        "percent_return": percent_return,
        "price_history": stock_data[['Close']]
    }