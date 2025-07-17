import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh

# --- Tickers to Monitor ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"]

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard")

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=30, max_value=300, value=60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# Define strategy categories first
bullish_strategies = [
    "Trend Trading", 
    "MACD Bullish Crossover", 
    "RSI Oversold", 
    "Bollinger Breakout"
]

bearish_strategies = [
    "MACD Bearish Crossover", 
    "RSI Overbought", 
    "Bollinger Rejection"
]

# Then use two multiselects independently
selected_bullish = st.sidebar.multiselect("📈 Bullish Strategies", bullish_strategies)
selected_bearish = st.sidebar.multiselect("📉 Bearish Strategies", bearish_strategies)


# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("""
**Trend Trading**: Shows uptrend signals when 20MA > 50MA  
**RSI Overbought**: Flags stocks with RSI > 70 for possible pullback  
**RSI Oversold**: Flags stocks with RSI < 30 for potential bounce  
**MACD Bullish Crossover**: Fast MACD line crosses above Signal line using (3, 10, 16)  
**MACD Bearish Crossover**: Fast MACD line crosses below Signal line using (3, 10, 16)  
**Bollinger Breakout**: Price closes above upper Bollinger Band, signaling potential breakout  
**Bollinger Rejection**: Price touches upper Bollinger Band and closes below it, signaling reversal
""")

# --- Data Processing & Signal Generation ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=5)
end = now

signals = []

for ticker in TICKERS:
    st.subheader(f"📈 {ticker}")
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")

        if df.empty or 'Close' not in df:
            st.warning(f"⚠️ No valid data for {ticker}.")
            continue

        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD with fast settings
        exp1 = df['Close'].ewm(span=3, adjust=False).mean()
        exp2 = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=16, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

bullish_strategies = [
    "Trend Trading", 
    "MACD Bullish Crossover", 
    "RSI Oversold", 
    "Bollinger Breakout"
]

bearish_strategies = [
    "MACD Bearish Crossover", 
    "RSI Overbought", 
    "Bollinger Rejection"
]

selected_bullish = st.sidebar.multiselect("📈 Bullish Strategies", bullish_strategies)
selected_bearish = st.sidebar.multiselect("📉 Bearish Strategies", bearish_strategies)


    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")

# --- Display Signals ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, msg in signals:
        st.success(msg)
else:
    st.info("No trade signals at this time.")
