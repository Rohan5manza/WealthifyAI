import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import backtrader as bt
import numpy as np
import openai
from bs4 import BeautifulSoup
import requests
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries


ALPHA_VANTAGE_API_KEY = "HYU5PH3HBAI9HV89"

# Initialize Alpha Vantage clients
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

#global function code
def get_ai_recommendations(ticker):
    try:
        # Get time series data
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Calculate SMA
        data['SMA50'] = data['4. close'].rolling(window=50).mean()
        data['SMA200'] = data['4. close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = data['4. close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get fundamental data
        overview, _ = fd.get_company_overview(symbol=ticker)
        
        # Analyze the data
        current_price = data['4. close'].iloc[-1]
        sma_50 = data['SMA50'].iloc[-1]
        sma_200 = data['SMA200'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        pe_ratio = float(overview.get('PERatio', 0))
        
        # Make a recommendation
        if sma_50 > sma_200 and current_rsi < 70 and pe_ratio > 0 and pe_ratio < 25:
            recommendation = "Buy"
        elif sma_50 < sma_200 and current_rsi > 30:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        
        return {
            'recommendation': recommendation,
            'current_price': current_price,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': current_rsi,
            'pe_ratio': pe_ratio
        }
    except Exception as e:
        st.error(f"Error getting AI recommendations for {ticker}: {str(e)}")
        return None



def set_page(page_name):
    st.session_state["page"] = page_name

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "Page 1"


# Navigation buttons
def render_navigation():
    st.sidebar.title("Navigation")
    if st.session_state["page"] != "Page 1":
        if st.sidebar.button("Go to Page 1"):
            set_page("Page 1")
    if st.session_state["page"] != "Page 2":
        if st.sidebar.button("Go to Page 2"):
            set_page("Page 2")
    if st.session_state["page"] != "Page 3":
        if st.sidebar.button("Go to Page 3"):
            set_page("Page 3")



def page_1():
# Helper function to get real-time stock data
    def get_stock_data(ticker, period='1d', interval='1m'):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data

    def Sharpe(ticker,period='1d',interval='1m'):
        
        data=get_stock_data(ticker,period,interval)      # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        
        # Drop missing data
        returns = data['Returns'].dropna()
        
        # Calculate average and standard deviation of returns
        average_return = returns.mean()
        std_dev_return = returns.std()
        risk_free_rate = 0.01  # Example placeholder
        # Calculate Sharpe Ratio
        sharpe_ratio = (average_return - risk_free_rate) / std_dev_return if std_dev_return != 0 else None    

        return sharpe_ratio
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

    # Pairs trading strategy class
    class PairsTrading(bt.Strategy):
        params = (('sma', 30),)
        
        def __init__(self):
            self.sma1 = bt.indicators.SimpleMovingAverage(self.data0.close, period=self.params.sma)
            self.sma2 = bt.indicators.SimpleMovingAverage(self.data1.close, period=self.params.sma)
            self.spread = self.data0.close - self.data1.close
            self.mean_spread = bt.indicators.SimpleMovingAverage(self.spread, period=30)
            self.std_spread = bt.indicators.StandardDeviation(self.spread, period=30)
            self.z_score = (self.spread - self.mean_spread) / self.std_spread
            self.orders = []
            self.trades = []  # Initialize trades as an empty list
            
        def next(self):
            if self.z_score[0] > 1:  # Spread is too wide, short stock 1, long stock 2
                if self.position.size == 0:
                    self.sell(data=self.data0)
                    self.buy(data=self.data1)
            elif self.z_score[0] < -1:  # Spread is too narrow, long stock 1, short stock 2
                if self.position.size == 0:
                    self.buy(data=self.data0)
                    self.sell(data=self.data1)
            elif -0.5 < self.z_score[0] < 0.5:  # Close both positions when spread normalizes
                self.close(data=self.data0)
                self.close(data=self.data1)
        
        def notify_trade(self, trade):
            if trade.isclosed:
                self.trades.append(trade)  # Append the trade object
                st.write(f"Trade closed: PnL: {trade.pnl}, Stock: {trade.data._name}")


    # Function to run backtest
    import numpy as np

    # Updated function to calculate Sharpe Ratio and other metrics
    def run_backtest(ticker1, ticker2, start_date='2020-01-01', end_date='2021-01-01'):
        cerebro = bt.Cerebro()
        
        # Download data
        data1 = yf.download(ticker1, start=start_date, end=end_date)
        data2 = yf.download(ticker2, start=start_date, end=end_date)
        
        # Add data to backtrader
        data1_bt = bt.feeds.PandasData(dataname=data1)
        data2_bt = bt.feeds.PandasData(dataname=data2)
        
        cerebro.adddata(data1_bt)
        cerebro.adddata(data2_bt)
        cerebro.addstrategy(PairsTrading)
        results = cerebro.run()
        
        # Calculate metrics
            # Assuming there is only one strategy in the results
        # Assuming there is only one strategy in the results
        strategy = results[0]
        
        # Initialize variables to track results
        total_profit = 0
        num_trades = len(strategy.trades)
        
        if num_trades > 0:
            # Calculate total profit
            total_profit = np.sum([trade.pnl for trade in strategy.trades if trade.isclosed])
            
            # Calculate average profit
            avg_profit = total_profit / num_trades
        else:
            avg_profit = 0
        # Get Sharpe Ratio
        stock_data1, sharpe_ratio1 = get_stock_data(ticker1, period='1y', interval='1d'), Sharpe(ticker1, period='1y', interval='1d')
        stock_data2, sharpe_ratio2 = get_stock_data(ticker2, period='1y', interval='1d'), Sharpe(ticker2, period='1y', interval='1d')
        
        sharpe_ratio = (sharpe_ratio1 + sharpe_ratio2) / 2 if sharpe_ratio1 is not None and sharpe_ratio2 is not None else None

        approval = {
            'sharpe_ratio': sharpe_ratio,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'num_trades': num_trades
        }
        
        return approval


    # Helper function to calculate technical indicators
    def calculate_technical_indicators(df):
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volume Drying Indicator
        df['Volume_Drying'] = df['Volume'].rolling(window=20).mean()

        return df

    # Helper function to identify buy/sell signals
    def generate_signals(df):
        df['Signal'] = 0
        df['Signal'][df['EMA_12'] > df['EMA_26']] = 1  # Buy Signal
        df['Signal'][df['EMA_12'] < df['EMA_26']] = -1  # Sell Signal
        
        df['Buy_Signal'] = (df['Signal'] == 1) & (df['RSI'] < 30) & (df['Close'] > df['SMA_50'])
        df['Sell_Signal'] = (df['Signal'] == -1) & (df['RSI'] > 70) & (df['Close'] < df['SMA_50'])
        
        return df
  

    # Main dashboard interface
    st.title("WealthifyAI")
    st.subheader("Technical Analysis and Quant strategies")
    # User input for stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT)")
    period = st.selectbox("Select Period", ["1d", "1wk", "1mo", "ytd", "1y", "5y", "max"])
    interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

    if ticker:
        # Display stock data
        st.subheader(f"Advanced technical analysis for {ticker}")
        
        stock_data = get_stock_data(ticker, period, interval)
        
        # Calculate technical indicators
        stock_data = calculate_technical_indicators(stock_data)
        stock_data = generate_signals(stock_data)
        # Plotting price with indicators
        st.subheader("Price and Indicators")
        fig = go.Figure()
        
        # Price and Moving Averages
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_200'], mode='lines', name='SMA 200'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_12'], mode='lines', name='EMA 12'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_26'], mode='lines', name='EMA 26'))
        
        # Buy/Sell signals
        buy_signals = stock_data[stock_data['Buy_Signal']]
        sell_signals = stock_data[stock_data['Sell_Signal']]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))
        
        fig.update_layout(title=f"{ticker} Technical Analysis", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
        
        # RSI Plot
        st.subheader("RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold', annotation_position='bottom right')
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought', annotation_position='top right')
        fig_rsi.update_layout(title=f"{ticker} RSI", xaxis_title='Date', yaxis_title='RSI')
        st.plotly_chart(fig_rsi)
        
        # Volume Plot
        st.subheader("Volume Drying Indicator")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'))
        fig_vol.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Volume_Drying'], mode='lines', name='Volume Drying'))
        fig_vol.update_layout(title=f"{ticker} Volume Drying Indicator", xaxis_title='Date', yaxis_title='Volume')
        st.plotly_chart(fig_vol)
        

        # Decision Making
        st.subheader("Trading Recommendations")
        buy_signals = stock_data[stock_data['Buy_Signal']]
        sell_signals = stock_data[stock_data['Sell_Signal']]
        
        if not buy_signals.empty:
            st.write("### Buy Recommendations")
            st.write(buy_signals[['Close', 'RSI', 'EMA_12', 'EMA_26', 'SMA_50']].tail())
        
        if not sell_signals.empty:
            st.write("### Sell Recommendations")
            st.write(sell_signals[['Close', 'RSI', 'EMA_12', 'EMA_26', 'SMA_50']].tail())
        
        # Trailing Stop Loss
        st.subheader("Trailing Stop Loss")
        st.write("Consider placing a trailing stop loss to protect profits. For a long position, set a trailing stop loss below the most recent high; for a short position, set it above the most recent low.")
    
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
        st.subheader("Factor Investing Strategy")
        st.write(""" Factor investing is a popular quant strategy that considers volume, momentum,volatility, and statistics of the stock, like P/E ratio, etc. Read decision column to understand when to buy or sell""")
        st.write(df_with_factors[['Close', 'Factor_Score', 'Decision']].tail(5))

        # Pairs trading backtest section
        st.subheader("Statistical Arbitrage Strategy")
        # Explanation for Pairs Trading
        st.subheader("What is Pairs Trading?")
        st.write("""
        Pairs trading is a trading strategy that involves simultaneously buying one stock and selling another stock in the same sector or with a high historical correlation. 
        The idea is to profit from the relative price movements of the two stocks. More correlation means more accuracy, more dependence on each other, and similar asset, 
        sector, or commodity. To check if this strategy is profitable for your chosen stocks, we use backtesting. 
        You can use this strategy to buy two stocks at the same time, and maximize profits based on contradictory price movements.
        **Example**: AAPL is highly correlated to MSFT, as both belong to the same sector and are similar industries.
        """)

        ticker1 = st.text_input("Enter First Stock Ticker for Pair Trading( the one you originally selected.)")
        ticker2 = st.text_input("Enter Second Stock Ticker for Pair Trading")
        if st.button("Run Pairs Trading Backtest"):
        
            if ticker1 and ticker2:
             approval = run_backtest(ticker1, ticker2)

        # Display results safely
            total_profit = approval.get('total_profit', 0)
            avg_profit = approval.get('avg_profit', 0)
            num_trades = approval.get('num_trades', 0)
            sharpe_ratio = approval.get('sharpe_ratio', 0)  # Default to 0 if None

            st.subheader("Backtest Results")
            st.write(f"Total Profit: ${total_profit:.2f}")
            st.write(f"Average Profit per Trade: ${avg_profit:.2f}")
            st.write(f"Number of Trades: {num_trades}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            # Approval decision
            if sharpe_ratio > 1 and total_profit > 0:
                st.success("The strategy is approved for trading!")
            else:
                st.warning("The strategy needs improvements.Choose different stock pairs for max profits")
            
        ai_recommendation = get_ai_recommendations(ticker)
        if ai_recommendation:
            st.subheader("AI-Powered Recommendation")
            st.write(f"Recommendation: {ai_recommendation['recommendation']}")
            st.write(f"50-day SMA: {ai_recommendation['sma_50']:.2f}")
            st.write(f"200-day SMA: {ai_recommendation['sma_200']:.2f}")
            st.write(f"RSI: {ai_recommendation['rsi']:.2f}")
            st.write(f"P/E Ratio: {ai_recommendation['pe_ratio']:.2f}")



    
    



#Page 2 : fundamental analysis
from scipy.optimize import minimize
import matplotlib.pyplot as plt
def page_2(): 

    st.title("Fundamental Analysis")
    st.write("Trading can be fradulent. Always verify stocks for the validity of a business before investing in it. Comply with SEBI regulations as needed.")
    st.subheader("Portfolio Optimization with Risk Parity")
    def get_stock_data(tickers, start_date='2020-01-01', end_date='2021-01-01'):
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=start_date, end=end_date)['Close']
        return pd.DataFrame(data)

# Helper function to calculate the portfolio risk
    def portfolio_risk(weights, covariance_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    # Objective function for risk parity
    def risk_parity_objective(weights, covariance_matrix):
        portfolio_risk_value = portfolio_risk(weights, covariance_matrix)
        marginal_risks = np.dot(covariance_matrix, weights) / portfolio_risk_value
        risk_contributions = weights * marginal_risks
        return np.sum((risk_contributions - np.mean(risk_contributions)) ** 2)

    # Function to calculate optimal weights using risk parity
    def calculate_risk_parity_weights(returns):
        covariance_matrix = returns.cov()
        num_assets = len(returns.columns)
        initial_weights = np.ones(num_assets) / num_assets
        
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1)] * num_assets
        
        result = minimize(risk_parity_objective, initial_weights, args=(covariance_matrix,), 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    # Function to compute portfolio returns
    def calculate_portfolio_returns(weights, returns):
        return np.dot(returns.mean(), weights)

    # Main dashboard interface
    

    # User input for stock tickers
    tickers = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime(2021, 1, 1))

    if tickers:
        tickers = [ticker.strip() for ticker in tickers.split(',')]
        stock_data = get_stock_data(tickers, start_date, end_date)
        
        # Calculate returns
        returns = stock_data.pct_change().dropna()
        
        # Calculate risk parity weights
        optimal_weights = calculate_risk_parity_weights(returns)
        
        # Portfolio metrics
        portfolio_return = calculate_portfolio_returns(optimal_weights, returns)
        portfolio_risk = portfolio_risk(optimal_weights, returns.cov())
        
        st.subheader("Optimal Weights")
        st.write("Allocate the following percentages to each stock in your portfolio to trade with minimum risks:")
        st.write(pd.DataFrame({'Ticker': tickers, 'Weight': optimal_weights}))
        
        st.subheader("Portfolio Metrics")
        st.write(f"Expected Annual Return: {portfolio_return * 252:.2%}")
        st.write(f"Portfolio Risk (Standard Deviation): {portfolio_risk * np.sqrt(252):.2%}")

        # Plot stock data
        st.subheader("Stock Prices")
        fig, ax = plt.subplots()
        stock_data.plot(ax=ax)
        plt.title("Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(fig)

        # Plot portfolio performance
        st.subheader("Portfolio Performance")
        cumulative_returns = (returns + 1).cumprod()
        portfolio_performance = (returns.dot(optimal_weights) + 1).cumprod()
        
        fig, ax = plt.subplots()
        cumulative_returns.plot(ax=ax, label='Individual Stocks')
        portfolio_performance.plot(ax=ax, label='Risk Parity Portfolio', linestyle='--')
        plt.title("Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        st.pyplot(fig)

    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

    def calculate_portfolio_optimization(df):
        mean_returns = expected_returns.mean_historical_return(df)
        cov_matrix = risk_models.sample_cov(df)
        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.max_sharpe()
        clean_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=True)
        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(clean_weights, latest_prices, total_portfolio_value=15000)
        allocation, leftover = da.lp_portfolio()
        return clean_weights, performance, allocation, leftover

    def fetch_fundamental_data(tickers):
        fundamental_data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            fundamental_data[ticker] = {
                'Info': stock.info,
                'Income Statement': stock.financials,
                'Quarterly Income Statement': stock.quarterly_financials,
                'Balance Sheet': stock.balance_sheet,
                'Quarterly Balance Sheet': stock.quarterly_balance_sheet,
                'Cash Flow': stock.cashflow,
                'Quarterly Cash Flow': stock.quarterly_cashflow,
                'Recommendations': stock.recommendations_summary,
                'News': stock.news
            }
        return fundamental_data

    def display_fundamental_data(fundamental_data):
        for ticker, data in fundamental_data.items():
            st.subheader(f"Fundamental Data for {ticker}")
            
            st.write("### Info")
            st.json(data['Info'])
            
            st.write("### Income Statement")
            st.dataframe(data['Income Statement'])
            
            st.write("### Quarterly Income Statement")
            st.dataframe(data['Quarterly Income Statement'])
            
            st.write("### Balance Sheet")
            st.dataframe(data['Balance Sheet'])
            
            st.write("### Quarterly Balance Sheet")
            st.dataframe(data['Quarterly Balance Sheet'])
            
            st.write("### Cash Flow")
            st.dataframe(data['Cash Flow'])
            
            st.write("### Quarterly Cash Flow")
            st.dataframe(data['Quarterly Cash Flow'])
            
            st.write("### Recommendations")
            st.dataframe(data['Recommendations'])
            
            st.write("### News")
            st.write(data['News'])
    
    if tickers:
        
        # Portfolio Optimization
        df = stock_data
        weights, performance, allocation, leftover = calculate_portfolio_optimization(df)
        
        st.subheader("Portfolio Optimization")
        st.write(" Assuming your total initial investment is INR 15000")
        st.write("Optimal Weights:", weights)
        st.write("Portfolio Performance:", performance)
        st.write("Discrete Allocation:", allocation)
        st.write("Funds Remaining:", f"INR{leftover:.2f}")

        # Fundamental Analysis
        fundamental_data = fetch_fundamental_data(tickers)
        display_fundamental_data(fundamental_data)
        
        tickers_list = [ticker.strip() for ticker in tickers]
        ai_recommendations = {}
        for ticker in tickers_list:
            ai_recommendation = get_ai_recommendations(ticker)
            if ai_recommendation:
                ai_recommendations[ticker] = ai_recommendation
        
        st.subheader("AI-Powered Portfolio Recommendations")
        for ticker, recommendation in ai_recommendations.items():
            st.write(f"{ticker}: {recommendation['recommendation']}")


from datetime import datetime, timedelta


def page_3():
    st.title("Sector Rotation Strategy")
    
    st.write("""
    Sector rotation is a powerful strategy to maximize profits from trading. Different sectors perform well at different times based on economic cycles, interest rates, and market sentiment. 
    This tool will help you analyze sector performance and make informed decisions about sector rotation.
    """)

    # Helper function to get stock data
    def get_stock_data(ticker, period='1y', interval='1d'):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data

    # Helper function to find related tickers
    def find_related_tickers(ticker):
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', '')
            industry = stock.info.get('industry', '')
            
            if not sector or not industry:
                return []
            
            # Use the sector ETF as a proxy for related stocks
            sector_etfs = {
                'Technology': 'XLK',
                'Financial': 'XLF',
                'Healthcare': 'XLV',
                'Consumer Cyclical': 'XLY',
                'Industrials': 'XLI',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Basic Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }
            
            etf = sector_etfs.get(sector, '')
            if not etf:
                return []
            
            etf_data = yf.Ticker(etf)
            top_holdings = etf_data.info.get('holdings', [])
            related_tickers = [holding['symbol'] for holding in top_holdings[:5]]
            related_tickers.append(ticker)  # Include the original ticker
            return list(set(related_tickers))  # Remove duplicates
        except Exception as e:
            st.error(f"Error finding related tickers: {e}")
            return []

    # Helper function to calculate sector performance based on related tickers
    def calculate_sector_performance(ticker):
        related_tickers = find_related_tickers(ticker)
        if not related_tickers:
            st.warning("No related tickers found for the given stock.")
            return None

        sector_performance = {}
        for related_ticker in related_tickers:
            try:
                data = get_stock_data(related_ticker)
                sector_performance[related_ticker] = data['Close'].pct_change().mean() * 100  # Average daily return in percentage
            except Exception as e:
                st.warning(f"Could not retrieve data for {related_ticker}: {e}")

        return sector_performance

    # Helper function to fetch market news
    def fetch_market_news():
        try:
            url = 'https://finviz.com/news.ashx'
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [item.text for item in soup.find_all('a', class_='nn-tab-link')]
            return headlines[:10]  # Return top 10 headlines
        except Exception as e:
            st.error(f"Error fetching market news: {e}")
            return []

    # Helper function to get economic indicators
    def get_economic_indicators():
        indicators = {
            '^TNX': 'Treasury Yield',
            '^VIX': 'Volatility Index',
            'CL=F': 'Crude Oil',
            'GC=F': 'Gold',
            'EURUSD=X': 'EUR/USD'
        }
        data = {}
        for symbol, name in indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                data[name] = info.get('regularMarketPrice', 'N/A')
            except Exception as e:
                st.error(f"Error fetching {name}: {e}")
                data[name] = 'N/A'
        return data

    # Main dashboard interface
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)")

    if ticker:
        st.subheader(f"Sector Performance Analysis for {ticker}")

        sector_performance = calculate_sector_performance(ticker)

        if sector_performance:
            st.write("### Sector Performance")
            performance_df = pd.DataFrame(list(sector_performance.items()), columns=['Ticker', 'Average Daily Return (%)'])
            performance_df = performance_df.sort_values('Average Daily Return (%)', ascending=False)
            st.dataframe(performance_df)

            # Plotting sector performance
            fig = go.Figure()
            fig.add_trace(go.Bar(x=performance_df['Ticker'], y=performance_df['Average Daily Return (%)'], name='Sector Performance'))
            fig.update_layout(title='Sector Performance Comparison', xaxis_title='Ticker', yaxis_title='Average Daily Return (%)')
            st.plotly_chart(fig)

        # Fetch and display market news
        st.subheader("Latest Market News")
        news_headlines = fetch_market_news()
        for headline in news_headlines:
            st.write(f"• {headline}")

        # Display economic indicators
        st.subheader("Current Economic Indicators")
        indicators = get_economic_indicators()
        for name, value in indicators.items():
            st.write(f"{name}: {value}")

        # Recommendations based on sector performance
        if sector_performance:
            st.subheader("Investment Recommendations")
            avg_sector_performance = np.mean(list(sector_performance.values()))
            sector_sentiment = "positive" if avg_sector_performance > 0 else "negative"
            
            st.write(f"The overall sentiment for the {ticker}'s sector is {sector_sentiment}.")
            
            if sector_sentiment == "positive":
                st.write(f"Consider increasing your allocation to the {ticker}'s sector. The following stocks in this sector are performing well:")
                top_performers = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)[:3]
                for stock, performance in top_performers:
                    st.write(f"• {stock}: {performance:.2f}% average daily return")
            else:
                st.write(f"The {ticker}'s sector is underperforming. Consider these alternatives:")
                st.write("• Defensive sectors like Utilities or Consumer Staples")
                st.write("• Look for opportunities in counter-cyclical sectors")
                st.write("• Consider increasing cash allocation or moving to fixed income")

            # Additional considerations based on economic indicators
            if indicators['Treasury Yield'] != 'N/A' and float(indicators['Treasury Yield']) > 3:
                st.write("• With high treasury yields, consider increasing allocation to financial sector stocks")
            if indicators['Volatility Index'] != 'N/A' and float(indicators['Volatility Index']) > 20:
                st.write("• High market volatility detected. Consider defensive strategies or hedging")
            if indicators['Crude Oil'] != 'N/A' and float(indicators['Crude Oil']) > 80:
                st.write("• High oil prices may benefit energy sector stocks")

        st.write("\nRemember: This analysis is based on historical data and current market indicators. Always do your own research and consider consulting with a financial advisor before making investment decisions.")
    
        sector_performance = calculate_sector_performance(ticker)
        if sector_performance:
            ai_recommendations = {}
            for related_ticker in sector_performance.keys():
                ai_recommendation = get_ai_recommendations(related_ticker)
                if ai_recommendation:
                    ai_recommendations[related_ticker] = ai_recommendation
            
            st.subheader("AI-Powered Sector Recommendations")
            for related_ticker, recommendation in ai_recommendations.items():
                st.write(f"{related_ticker}: {recommendation['recommendation']}")


def page_4():

    st.title("AI insights,recommendations,and predictions")    
    # LLM ROBOADVISOR 



#main function for pages logic
def main():
    render_navigation()  # Sidebar navigation

    # Display the current page based on session state
    if st.session_state["page"] == "Page 1":
        page_1()
    elif st.session_state["page"] == "Page 2":
        page_2()
    elif st.session_state["page"] == "Page 3":
        page_3()
    
# Start the app
if __name__ == "__main__":
    main()