import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import sendgrid
from sendgrid.helpers.mail import Mail
import pandas as pd

# Setting up APIs
NEWS_API_KEY = "your_newsapi_key"
SENDGRID_API_KEY = "your_sendgrid_key"

# Helper function to get real-time stock data
def get_stock_data(ticker, period='1d', interval='1m'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Helper function to calculate factor investing scores
def calculate_factor_scores(df):
    df['PE_Ratio'] = df['Close'] / df['Close'].mean()  # Example placeholder
    df['Momentum'] = df['Close'].pct_change(periods=10)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    df['Value_Score'] = -df['PE_Ratio'].fillna(0)
    df['Momentum_Score'] = df['Momentum'].fillna(0)
    df['Volatility_Score'] = -df['Volatility'].fillna(0)
    df['Factor_Score'] = 0.4 * df['Value_Score'] + 0.3 * df['Momentum_Score'] + 0.3 * df['Volatility_Score']
    df['Decision'] = df['Factor_Score'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    return df

# Helper function for statistical arbitrage (pairs trading example)
def pairs_trading_strategy(df1, df2):
    # Merge dataframes on date
    df = pd.merge(df1[['Close']], df2[['Close']], left_index=True, right_index=True, suffixes=('_1', '_2'))
    df['Spread'] = df['Close_1'] - df['Close_2']
    df['Mean_Spread'] = df['Spread'].rolling(window=30).mean()
    df['Std_Spread'] = df['Spread'].rolling(window=30).std()
    df['Z_Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    
    # Trading signals
    df['Long_Signal'] = df['Z_Score'] < -1
    df['Short_Signal'] = df['Z_Score'] > 1
    
    return df


# Main dashboard interface
st.title("Interactive Trading Dashboard")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT)")
period = st.selectbox("Select Period", ["1d", "1wk", "1mo", "ytd", "1y", "5y", "max"])
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

if ticker:
    # Display stock data
    st.subheader(f"Stock Data for {ticker}")
    
    stock_data = get_stock_data(ticker, period, interval)
    st.write(stock_data.tail(5))
    
    # Calculate factor scores and decisions
    df_with_factors = calculate_factor_scores(stock_data)
    
    # Plotting real-time stock price graph
    st.subheader("Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
    fig.update_layout(title=f"{ticker} Price", xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
    # Plotting candlestick chart
    st.subheader("Candlestick Chart")
    fig_candlestick = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                     open=stock_data['Open'],
                                                     high=stock_data['High'],
                                                     low=stock_data['Low'],
                                                     close=stock_data['Close'])])
    fig_candlestick.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig_candlestick)
    
    # Display factor investing decisions
    st.subheader("Factor Investing Decisions")
    st.write(df_with_factors[['Close', 'Factor_Score', 'Decision']].tail(5))
    
import backtrader as bt


# Define a strategy
class PairsTrading(bt.Strategy):
    params = (('sma', 30),)
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma)
        self.order = None
    
    def next(self):
        if self.order:
            return
        
        if self.data.close[0] > self.sma[0]:
            self.sell()
        elif self.data.close[0] < self.sma[0]:
            self.buy()

# Function to run backtest
def run_backtest(ticker1, ticker2, start_date='2020-01-01', end_date='2021-01-01'):
    cerebro = bt.Cerebro()
    
    # Download data
    data1 = bt.feeds.PandasData(dataname=yf.download(ticker1, start=start_date, end=end_date))
    data2 = bt.feeds.PandasData(dataname=yf.download(ticker2, start=start_date, end=end_date))
    
    cerebro.adddata(data1)
    cerebro.adddata(data2)
    cerebro.addstrategy(PairsTrading)
    results = cerebro.run()
    
    # Extract performance metrics
    portfolio_value = cerebro.broker.getvalue()
    trade_list = results[0].trades
    num_trades = len(trade_list)
    profits = [trade.pnl for trade in trade_list]
    total_profit = np.sum(profits)
    avg_profit = np.mean(profits)
    
    # Calculate Sharpe Ratio
    returns = np.array(profits) / portfolio_value
    risk_free_rate = 0.01
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
    
    # Approval criteria
    approval = {
        'sharpe_ratio': sharpe_ratio,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'num_trades': num_trades
    }
    
    return approval

ticker1 = st.text_input("Enter First Stock Ticker for Pair Trading")
ticker2 = st.text_input("Enter Second Stock Ticker for Pair Trading")
if st.button("Run Pairs Trading Backtest"):
    if ticker1 and ticker2:
        approval = run_backtest(ticker1, ticker2)
        st.write("Backtest completed!")
        
        # Display performance metrics
        st.subheader("Backtest Results")
        st.write(f"Total Profit: ${approval['total_profit']:.2f}")
        st.write(f"Average Profit per Trade: ${approval['avg_profit']:.2f}")
        st.write(f"Number of Trades: {approval['num_trades']}")
        st.write(f"Sharpe Ratio: {approval['sharpe_ratio']:.2f}")
        
        # Approval decision
        if approval['sharpe_ratio'] > 1 and approval['total_profit'] > 0:
            st.success("The strategy is approved based on the backtest results!")
        else:
            st.error("The strategy did not meet the approval criteria based on the backtest results.")
    else:
        st.error("Please enter both stock tickers.")
