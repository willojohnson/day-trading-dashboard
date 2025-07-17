import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh

# --- Tickers to Monitor ---
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", 
    "BBAI", "SOUN", "CRSP", "TSM", "DDOG"
]

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard")

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=30, max_value=300, value=60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

strategy = st.sidebar.selectbox(
    "Select Strategy", [
        "Trend Trading", 
        "RSI Overbought", 
        "RSI Oversold", 
        "MACD Bullish Crossover", 
        "MACD Bearish Crossover", 
        "Bollinger Breakout", 
        "Bollinger Rejection"
    ]
)

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

        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data for {ticker}.")
            continue

        # Moving averages
        df['20_MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['50_MA'] = df['Close'].rolling(window=50, min_periods=1).mean()

        # Standard RSI calculation
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14, min_periods=1).mean()
        roll_down = down.rolling(14, min_periods=1).mean()
        rs = roll_up / (roll_down.replace(0, 1e-8))
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (3, 10, 16)
        exp1 = df['Close'].ewm(span=3, adjust=False).mean()
        exp2 = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=16, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

        # Defensive: require at least two bars for crossovers, etc
        if len(df) < 2:
            st.warning(f"Not enough data for {ticker} to compute signals.")
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if strategy == "Trend Trading":
            if pd.notna(last['20_MA']) and pd.notna(last['50_MA']) and last['20_MA'] > last['50_MA']:
                signal = f"📈 Trend: {ticker} 20MA > 50MA"
                signals.append((ticker, signal))

        elif strategy == "RSI Overbought":
            if pd.notna(last['RSI']) and last['RSI'] > 70:
                signal = f"🔺 RSI Overbought: {ticker} RSI={last['RSI']:.1f}"
                signals.append((ticker, signal))

        elif strategy == "RSI Oversold":
            if pd.notna(last['RSI']) and last['RSI'] < 30:
                signal = f"🔻 RSI Oversold: {ticker} RSI={last['RSI']:.1f}"
                signals.append((ticker, signal))

        elif strategy == "MACD Bullish Crossover":
            if (
                pd.notna(prev['MACD']) and pd.notna(prev['MACD_Signal'])
                and pd.notna(last['MACD']) and pd.notna(last['MACD_Signal'])
            ):
                if prev['MACD'] < prev['MACD_Signal'] and last['MACD'] > last['MACD_Signal']:
                    signal = f"📊 MACD Bullish Crossover: {ticker}"
                    signals.append((ticker, signal))

        elif strategy == "MACD Bearish Crossover":
            if (
                pd.notna(prev['MACD']) and pd.notna(prev['MACD_Signal'])
                and pd.notna(last['MACD']) and pd.notna(last['MACD_Signal'])
            ):
                if prev['MACD'] > prev['MACD_Signal'] and last['MACD'] < last['MACD_Signal']:
                    signal = f"📉 MACD Bearish Crossover: {ticker}"
                    signals.append((ticker, signal))

        elif strategy == "Bollinger Breakout":
            if pd.notna(last['Close']) and pd.notna(last['BB_Upper']):
                if last['Close'] > last['BB_Upper']:
                    signal = f"🚀 Bollinger Breakout: {ticker} closed above upper band"
                    signals.append((ticker, signal))

        elif strategy == "Bollinger Rejection":
            if pd.notna(last['High']) and pd.notna(last['BB_Upper']) and pd.notna(last['Close']):
                if last['High'] > last['BB_Upper'] and last['Close'] < last['BB_Upper']:
                    signal = f"⚠️ Bollinger Rejection: {ticker} touched upper band and reversed"
                    signals.append((ticker, signal))
    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")

# --- Display Signals ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, msg in signals:
        st.success(msg)
else:
    st.info("No trade signals at this time.")
'''

# Place this button anywhere in your Streamlit sidebar or main view.
st.download_button(
    label="Download Corrected Dashboard Code (.py)",
    data=corrected_code,
    file_name="trading_dashboard.py",
    mime="text/x-python"
)
