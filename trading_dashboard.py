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

strategy = st.sidebar.selectbox("Select Strategy", ["Trend Trading", "RSI Overbought", "RSI Oversold", "MACD Crossover"])

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("""
**Trend Trading**: Shows uptrend signals when 20MA > 50MA  
**RSI Overbought**: Flags stocks with RSI > 70 for possible pullback  
**RSI Oversold**: Flags stocks with RSI < 30 for potential bounce  
**MACD Crossover**: Flags bullish crossovers using fast MACD settings (3, 10, 16)
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
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().add(1).rolling(14).apply(
            lambda x: (x[x > 1].mean() / x[x <= 1].mean()) if x[x <= 1].mean() != 0 else 1, raw=False)))

        # MACD with fast settings
        exp1 = df['Close'].ewm(span=3, adjust=False).mean()
        exp2 = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=16, adjust=False).mean()

        if strategy == "Trend Trading":
            if pd.notna(df['20_MA'].iloc[-1]) and pd.notna(df['50_MA'].iloc[-1]) and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
                signal = f"📈 Trend: {ticker} 20MA > 50MA"
                signals.append((ticker, signal))

        elif strategy == "RSI Overbought":
            if pd.notna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 70:
                signal = f"🔺 RSI Overbought: {ticker} RSI={df['RSI'].iloc[-1]:.1f}"
                signals.append((ticker, signal))

        elif strategy == "RSI Oversold":
            if pd.notna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] < 30:
                signal = f"🔻 RSI Oversold: {ticker} RSI={df['RSI'].iloc[-1]:.1f}"
                signals.append((ticker, signal))

        elif strategy == "MACD Crossover":
            if pd.notna(df['MACD'].iloc[-2]) and pd.notna(df['MACD_Signal'].iloc[-2]) and pd.notna(df['MACD'].iloc[-1]) and pd.notna(df['MACD_Signal'].iloc[-1]):
                if df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    signal = f"📊 MACD Bullish Crossover: {ticker}"
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
