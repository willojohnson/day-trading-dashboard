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

strategy = st.sidebar.selectbox("Select Strategy", ["Trend Trading", "RSI Overbought", "Scalping", "Breakout", "Lower High + Lower Low"])

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("""
**Trend Trading**: Shows uptrend signals when 20MA > 50MA  
**RSI Overbought**: Flags stocks with RSI > 70 for possible pullback  
**Scalping**: Short-term trades triggered by volume surges and 20MA > 50MA  
**Breakout**: Flags when current price breaks above 20-period high  
**Lower High + Lower Low**: Detects weakening trends when each bar has a lower high and lower low than the previous bar
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
        df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
        df['20_High'] = df['High'].rolling(window=20).max()

        if strategy == "Trend Trading":
            if pd.notna(df['20_MA'].iloc[-1]) and pd.notna(df['50_MA'].iloc[-1]) and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
                signal = f"📈 Trend: {ticker} 20MA > 50MA"
                signals.append((ticker, signal))

        elif strategy == "RSI Overbought":
            if pd.notna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 70:
                signal = f"🔺 RSI Overbought: {ticker} RSI={df['RSI'].iloc[-1]:.1f}"
                signals.append((ticker, signal))

        elif strategy == "Scalping":
            if pd.notna(df['20_MA'].iloc[-1]) and pd.notna(df['50_MA'].iloc[-1]) and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
                if df['Volume'].iloc[-1] > 1.5 * df['Avg_Volume'].iloc[-1]:
                    signal = f"⚡ Scalping: {ticker} volume surge & 20MA > 50MA"
                    signals.append((ticker, signal))

        elif strategy == "Breakout":
            required_columns = ['Close', '20_High']
            if all(col in df.columns for col in required_columns):
                if pd.notna(df['Close'].iloc[-1]) and pd.notna(df['20_High'].iloc[-2]):
                    if df['Close'].iloc[-1] > df['20_High'].iloc[-2]:
                        signal = f"🔹 Breakout: {ticker} price > 20-period high"
                        signals.append((ticker, signal))

        elif strategy == "Lower High + Lower Low":
            if len(df) >= 2:
                if (df['High'].iloc[-1] < df['High'].iloc[-2]) and (df['Low'].iloc[-1] < df['Low'].iloc[-2]):
                    signal = f"🔻 Lower High + Lower Low: {ticker} weakening trend"
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
