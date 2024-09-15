import yfinance as yf
import pandas as pd

def period_to_yf(period: str) -> str:
    mapping = {
        "1 day": "1d",
        "1 week": "1wk",
        "1 month": "1mo",
        "YTD": "ytd",
        "1 year": "1y",
        "5 years": "5y",
        "max": "max"
    }
    return mapping.get(period, "1y")

def get_stock_data(ticker: str, period="1y"):
    period = period_to_yf(period)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_factor_scores(df):
    # Example factors for demonstration
    df['PE_Ratio'] = df['Close'] / df['Close'].mean()  # Example placeholder
    df['Momentum'] = df['Close'].pct_change(periods=10)
    df['Volatility'] = df['Close'].rolling(window=20).std()

    df['Value_Score'] = -df['PE_Ratio'].fillna(0)
    df['Momentum_Score'] = df['Momentum'].fillna(0)
    df['Volatility_Score'] = -df['Volatility'].fillna(0)
    df['Factor_Score'] = 0.4 * df['Value_Score'] + 0.3 * df['Momentum_Score'] + 0.3 * df['Volatility_Score']
    df['Decision'] = df['Factor_Score'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    return df
