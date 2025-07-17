import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# --- Tickers to Monitor ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"]

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard")

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=30, max_value=300, value=60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Strategy Selectors ---
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

# --- Signal Generation ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=5)
end = now

signals = []
heatmap_data = []

for ticker in TICKERS:
    st.subheader(f"📈 {ticker}")
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")

        if df.empty or 'Close' not in df:
            st.warning(f"⚠️ No valid data for {ticker}.")
            continue

        # --- Indicators ---
        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['Close'].ewm(span=3, adjust=False).mean()
        exp2 = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=16, adjust=False).mean()
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

        # --- Signal Tracking ---
        heatmap_row = {"Ticker": ticker}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = 0

        # --- Bullish Signals ---
        if "Trend Trading" in selected_bullish:
            if pd.notna(df['20_MA'].iloc[-1]) and pd.notna(df['50_MA'].iloc[-1]) and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
                signals.append((ticker, f"📈 Bullish - Trend Trading"))
                heatmap_row["Trend Trading"] = 1

        if "RSI Oversold" in selected_bullish:
            if pd.notna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] < 30:
                signals.append((ticker, f"📈 Bullish - RSI Oversold (RSI={df['RSI'].iloc[-1]:.1f})"))
                heatmap_row["RSI Oversold"] = 1

        if "MACD Bullish Crossover" in selected_bullish:
            if all(pd.notna(val) for val in [
                df['MACD'].iloc[-2], df['MACD_Signal'].iloc[-2],
                df['MACD'].iloc[-1], df['MACD_Signal'].iloc[-1]]):
                if df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    signals.append((ticker, f"📈 Bullish - MACD Bullish Crossover"))
                    heatmap_row["MACD Bullish Crossover"] = 1

        if "Bollinger Breakout" in selected_bullish:
            last = df.tail(1)
            close = last['Close'].values[0]
            upper = last['BB_Upper'].values[0]
            if pd.notna(close) and pd.notna(upper) and close > upper:
                signals.append((ticker, f"📈 Bullish - Bollinger Breakout"))
                heatmap_row["Bollinger Breakout"] = 1

        # --- Bearish Signals ---
        if "RSI Overbought" in selected_bearish:
            if pd.notna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 70:
                signals.append((ticker, f"📉 Bearish - RSI Overbought (RSI={df['RSI'].iloc[-1]:.1f})"))
                heatmap_row["RSI Overbought"] = 1

        if "MACD Bearish Crossover" in selected_bearish:
            if all(pd.notna(val) for val in [
                df['MACD'].iloc[-2], df['MACD_Signal'].iloc[-2],
                df['MACD'].iloc[-1], df['MACD_Signal'].iloc[-1]]):
                if df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
                    signals.append((ticker, f"📉 Bearish - MACD Bearish Crossover"))
                    heatmap_row["MACD Bearish Crossover"] = 1

        if "Bollinger Rejection" in selected_bearish:
            last = df.tail(1)
            high = last['High'].values[0]
            close = last['Close'].values[0]
            upper = last['BB_Upper'].values[0]
            if pd.notna(high) and pd.notna(close) and pd.notna(upper) and high >= upper and close < upper:
                signals.append((ticker, f"📉 Bearish - Bollinger Rejection"))
                heatmap_row["Bollinger Rejection"] = 1

        heatmap_data.append(heatmap_row)

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")

# --- Display Signals ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, msg in signals:
        st.success(msg)
else:
    st.info("No trade signals at this time.")

# --- Display Signal Matrix & Visual Heatmap ---
if heatmap_data:
    st.markdown("### 🧭 Strategy Signal Matrix")

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df["Bullish Total"] = heatmap_df[[col for col in bullish_strategies if col in heatmap_df.columns]].sum(axis=1)
    heatmap_df["Bearish Total"] = heatmap_df[[col for col in bearish_strategies if col in heatmap_df.columns]].sum(axis=1)

    ordered_cols = ["Ticker"] + bullish_strategies + ["Bullish Total"] + bearish_strategies + ["Bearish Total"]
    heatmap_df = heatmap_df[ordered_cols]

    st.dataframe(heatmap_df.style.highlight
