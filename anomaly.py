import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data.dropna(inplace=True)
    return stock_data

def calculate_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data['BB_Upper'] = data['SMA_20'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['SMA_20'] - 2 * data['Close'].rolling(window=20).std()
    data['BB_Middle'] = data['SMA_20']
    data['Returns'] = data['Close'].pct_change()
    return data

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_anomalies(data):
    features = ['Close', 'Volume', 'Returns', 'RSI', 'SMA_20', 'EMA_20', 'BB_Upper', 'BB_Lower']
    X = data[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    data['Anomaly'] = 0
    data.loc[X.index, 'Anomaly'] = anomalies
    return data

def plot_anomalies(data, ticker):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    anomaly_data = data[data['Anomaly'] == -1]
    ax.scatter(anomaly_data.index, anomaly_data['Close'], color='red', label='Anomalies', marker='o')
    ax.set_title(f'{ticker} Stock Price with Anomalies')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig

def main():
    st.title("Stock Price Anomaly Detection")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    end_date = st.date_input("End Date", datetime.now())
    start_date = st.date_input("Start Date", end_date - timedelta(days=365))
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing data..."):
            stock_data = download_stock_data(ticker, start_date, end_date)
            stock_data = calculate_indicators(stock_data)
            stock_data = detect_anomalies(stock_data)
            st.subheader("Anomaly Detection Results")
            anomaly_count = len(stock_data[stock_data['Anomaly'] == -1])
            st.write(f"""
            **Anomaly Detection Report for {ticker}**
            - Period: {start_date} to {end_date}
            - Total Data Points: {len(stock_data)}
            - Detected Anomalies: {anomaly_count}
            - Anomaly Percentage: {(anomaly_count / len(stock_data)) * 100:.2f}%
            """)
            fig = plot_anomalies(stock_data, ticker)
            st.pyplot(fig)
            st.subheader("Data with Anomalies")
            st.dataframe(stock_data[stock_data['Anomaly'] == -1][['Close', 'Volume', 'RSI', 'Anomaly']])

if __name__ == "__main__":
    main()