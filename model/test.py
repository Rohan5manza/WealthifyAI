import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta
from PIL import Image
import requests
import sendgrid
from sendgrid.helpers.mail import Mail

# Set up APIs
NEWS_API_KEY = "your_newsapi_key"
SENDGRID_API_KEY = "your_sendgrid_key"

# Helper function to get stock data
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    if period == '1wk':
        data = stock.history(period='1wk', interval='1m')
    elif period == '1d':
        data = stock.history(period='1d', interval='1m')  # 1 day with 1-minute intervals
    else:
        data = stock.history(period=period)
    return data

# Helper function to get options data
def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    options = stock.options
    if options:
        return {option: stock.option_chain(option)._asdict() for option in options}
    return {}

# Helper function to get latest news
def get_latest_news(query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
    response = requests.get(url).json()
    return response['articles']

# Helper function to calculate trading indicators
def calculate_indicators(data):
    data = dropna(data)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return data

# Helper function to send email notifications
def send_email_notification(user_email, subject, content):
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    message = Mail(
        from_email='your_email@domain.com',
        to_emails=user_email,
        subject=subject,
        html_content=content)
    sg.send(message)

# Main dashboard interface
st.title("Interactive Trading Dashboard")


# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT)")

# Options for graph time ranges
time_range = st.selectbox(
    "Select Time Range",
    ["1d", "1wk", "1mo", "1y", "ytd", "5y", "max"]
)

if ticker:
    # Display stock data
    st.subheader(f"Stock Data for {ticker}")

    stock_data = get_stock_data(ticker, time_range)

    # Ensure stock_data is a DataFrame and has data
    if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
        st.write(stock_data.tail(5))

        # Calculate trading indicators
        indicators_data = calculate_indicators(stock_data)

        # Plotting stock price graph
        st.subheader("Price Movement")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
        fig.update_layout(title=f"{ticker} Price", xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig)

        # Plotting indicators
        st.subheader("Trading Indicators")

        # Check if indicator columns exist before plotting
        if 'trend_macd' in indicators_data.columns:
            st.write("**Moving Averages (MACD)**")
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=stock_data.index, y=indicators_data['trend_macd'], mode='lines', name='MACD'))
            fig_ma.update_layout(title=f"{ticker} MACD", xaxis_title='Time', yaxis_title='MACD')
            st.plotly_chart(fig_ma)

        if 'momentum_rsi' in indicators_data.columns:
            st.write("**Relative Strength Index (RSI)**")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=indicators_data['momentum_rsi'], mode='lines', name='RSI'))
            fig_rsi.update_layout(title=f"{ticker} RSI", xaxis_title='Time', yaxis_title='RSI')
            st.plotly_chart(fig_rsi)

        # Display options data if available
        st.subheader(f"Options Data for {ticker}")
        options_data = get_options_data(ticker)
        if options_data:
            st.write(options_data)
        else:
            st.write("No options data available.")

        # Display latest news
        st.subheader(f"Latest News for {ticker}")
        articles = get_latest_news(ticker)
        for article in articles[:5]:
            st.markdown(f"**[{article['title']}]({article['url']})** - {article['source']['name']}")
            st.write(article['description'])

        # Email notification button
        user_email = st.text_input("Enter your email to get notifications")
        if st.button("Send Email Notification"):
            subject = f"Stock Price Alert for {ticker}"
            content = f"The latest stock price for {ticker} is {stock_data['Close'].iloc[-1]}."
            send_email_notification(user_email, subject, content)
            st.success(f"Notification sent to {user_email}")
    else:
        st.error("No data available for the selected period.")
