import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import base64
import os
from collections import defaultdict
from streamlit_autorefresh import st_autorefresh
import socket

# --- AI Watchlist ---
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "PLTR", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"
]

# --- Leaderboard History ---
signal_leaderboard = defaultdict(int)

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("\U0001F4C8 Day Trading Dashboard")

bullish_strategies = ["Breakout", "Scalping", "Trend Trading"]
bearish_strategies = ["RSI Overbought", "Lower High + Lower Low", "Volume Spike Down", "Shooting Star"]

strategy_type = st.sidebar.radio("Strategy Type", ["Bullish", "Bearish"])
if strategy_type == "Bullish":
    strategy = st.sidebar.selectbox("Select Bullish Strategy", bullish_strategies)
elif strategy_type == "Bearish":
    strategy = st.sidebar.selectbox("Select Bearish Strategy", bearish_strategies)

refresh_rate = st.sidebar.slider("Refresh every N seconds", 30, 300, 60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# Strategy Definitions – Always Visible
st.sidebar.markdown("### \U0001F4D8 Strategy Definitions")
st.sidebar.markdown("""
**Breakout**  
Triggered when price breaks above recent highs.

**Scalping**  
Short trades triggered by volume surges and 20MA > 50MA.

**Trend Trading**  
20MA > 50MA means momentum likely continuing.

**RSI Overbought**  
RSI above 70 suggests a pullback.

**Lower High + Lower Low**  
Signals weakening uptrend and possible reversal.

**Volume Spike Down**  
Large red candle with volume spike.

**Shooting Star**  
Small body, long upper wick near intraday highs.
""")

# Manual Refresh Button
if st.sidebar.button("\U0001F501 Refresh Now"):
    st.rerun()

# Sound alert function
def play_alert():
    sound_file_path = "alert.mp3"
    if os.path.exists(sound_file_path):
        b64_sound = base64.b64encode(open(sound_file_path, "rb").read()).decode()
        sound_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_sound}" type="audio/mp3">
            </audio>"""
        st.markdown(sound_html, unsafe_allow_html=True)

# Create dummy alert file
with open("alert.mp3", "wb") as f:
    f.write(b"ID3\x03\x00\x00\x00\x00\x00!TIT2\x00\x00\x00\x07\x00\x00\x03Beep\x00\x00")

placeholder = st.empty()

with placeholder.container():
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=5)
    end_date = now
    ranked_signals = []

    for ticker in TICKERS:
        st.subheader(f"Loading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, interval="5m")
        if data.empty or len(data) < 50 or 'Close' not in data.columns:
            st.error(f"Not enough or invalid data for {ticker}.")
            continue

        data['High_Break'] = data['High'].rolling(window=20).max()
        data['Low_Break'] = data['Low'].rolling(window=20).min()
        data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5
        data['Momentum'] = data['Close'].pct_change().rolling(window=10).sum()
        data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().add(1).rolling(14).apply(lambda x: (x[x > 1].mean() / x[x <= 1].mean()) if x[x <= 1].mean() else 1)))

        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['50_MA'] = data['Close'].rolling(window=50).mean()

        signal = ""
        trade_flag = False
        rank_value = 0

        try:
            close = data['Close'].iloc[-1]
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            open_ = data['Open'].iloc[-1]
            prev_close = data['Close'].iloc[-2]

            if strategy == "Breakout":
                high_break = data['High_Break'].dropna()
                if not high_break.empty and pd.notna(close):
                    last_high = high_break.iloc[-1]
                    if pd.notna(last_high) and close > last_high:
                        signal = f"\U0001F514 Breakout: {ticker} above recent high"
                        trade_flag = True
                        rank_value = data['Momentum'].iloc[-1] if pd.notna(data['Momentum'].iloc[-1]) else 0

            elif strategy == "Scalping":
                if data['20_MA'].iloc[-1] > data['50_MA'].iloc[-1] and data['Volume_Surge'].iloc[-1]:
                    signal = f"⚡ Scalping: {ticker} volume surge & 20MA > 50MA"
                    trade_flag = True
                    rank_value = data['Volume'].iloc[-1]

            elif strategy == "Trend Trading":
                if '20_MA' in data.columns and '50_MA' in data.columns and not pd.isna(data['20_MA'].iloc[-1]) and not pd.isna(data['50_MA'].iloc[-1]):
                    if data['20_MA'].iloc[-1] > data['50_MA'].iloc[-1]:
                        signal = f"\U0001F4C8 Trend: {ticker} in uptrend (20MA > 50MA)"
                        trade_flag = True
                        rank_value = data['Momentum'].iloc[-1]

            elif strategy == "RSI Overbought":
                if data['RSI'].iloc[-1] > 70:
                    signal = f"\U0001F53B RSI Overbought: {ticker} RSI={data['RSI'].iloc[-1]:.1f}"
                    trade_flag = True
                    rank_value = -data['RSI'].iloc[-1]

            elif strategy == "Lower High + Lower Low":
                if data['High'].iloc[-1] < data['High'].iloc[-2] and data['Low'].iloc[-1] < data['Low'].iloc[-2]:
                    signal = f"🔻 Bearish Pattern: {ticker} lower high + lower low"
                    trade_flag = True
                    rank_value = -data['Momentum'].iloc[-1]

            elif strategy == "Volume Spike Down":
                avg_vol = data['Volume'].rolling(window=20).mean().iloc[-1]
                if data['Volume'].iloc[-1] > avg_vol * 1.5 and close < open_:
                    signal = f"📉 Volume Spike Down: {ticker} large red candle w/ high volume"
                    trade_flag = True
                    rank_value = -abs(data['Momentum'].iloc[-1])

            elif strategy == "Shooting Star":
                candle_body = abs(close - open_)
                upper_wick = high - max(close, open_)
                if upper_wick > candle_body * 2:
                    signal = f"🌠 Shooting Star: {ticker} — potential intraday reversal"
                    trade_flag = True
                    rank_value = -data['Momentum'].iloc[-1]

        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")

        if trade_flag:
            ranked_signals.append((ticker, signal, rank_value))
            signal_leaderboard[ticker] += 1
            play_alert()

    if ranked_signals:
        st.markdown("### \U0001F4CA Real-Time Signals")
        ranked_signals.sort(key=lambda x: x[2])
        for ticker, signal, rank in ranked_signals:
            st.success(signal)

    if signal_leaderboard:
        leaderboard_df = pd.DataFrame(sorted(signal_leaderboard.items(), key=lambda x: x[1], reverse=True), columns=['Ticker', 'Signal Count'])
        st.markdown("### \U0001F3C6 Signal Leaderboard")
        st.dataframe(leaderboard_df)

    environment = "Localhost" if socket.gethostname() == "localhost" else "Streamlit Cloud"
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({environment})")

    try:
        with open(__file__, "r", encoding="utf-8") as f:
            full_code = f.read()
    except:
        full_code = "# Source code not available in this environment."

    st.download_button(
        label="\U0001F4E5 Download Updated Script",
        data=full_code,
        file_name="trading_dashboard.py",
        mime="text/plain"
    )
