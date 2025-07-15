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
bearish_strategies = ["VWAP Rejection", "RSI Overbought", "Lower High + Lower Low", "Volume Spike Down", "Shooting Star", "VWAP Retest Fail"]

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
Triggered when price breaks above recent highs *and* trades above intraday VWAP.

**Scalping**  
Short trades triggered by volume surges and 20MA > 50MA.

**Trend Trading**  
20MA > 50MA means momentum likely continuing.

**VWAP Rejection**  
Price breaks above VWAP but closes below it.

**RSI Overbought**  
RSI above 70 suggests a pullback.

**Lower High + Lower Low**  
Signals weakening uptrend and possible reversal.

**Volume Spike Down**  
Large red candle with volume spike.

**Shooting Star**  
Small body, long upper wick near intraday highs.

**VWAP Retest Fail**  
Price reclaims VWAP briefly, then drops below.
""")

if st.sidebar.button("\U0001F501 Refresh Now"):
    st.rerun()

def play_alert():
    sound_file_path = "alert.mp3"
    if os.path.exists(sound_file_path):
        b64_sound = base64.b64encode(open(sound_file_path, "rb").read()).decode()
        sound_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_sound}" type="audio/mp3">
            </audio>"""
        st.markdown(sound_html, unsafe_allow_html=True)

with open("alert.mp3", "wb") as f:
    f.write(b"ID3\x03\x00\x00\x00\x00\x00\x21TIT2\x00\x00\x00\x07\x00\x00\x03Beep\x00\x00")

placeholder = st.empty()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

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

        if all(col in data.columns for col in ['High', 'Low', 'Close', 'Volume']):
            typical_price = ((data['High'] + data['Low'] + data['Close']) / 3).astype(float).fillna(0)
            volume = pd.to_numeric(data['Volume'], errors='coerce').astype(float).fillna(0)

            tpv = (typical_price * volume).fillna(0)
            cum_vol = volume.cumsum().replace(0, 1e-9)
            vwap = tpv.cumsum() / cum_vol
            data['VWAP'] = vwap.fillna(0)

        # Pre-calculate indicators, handling missing values robustly
        data['High_Break'] = data['High'].rolling(window=20).max()
        data['Low_Break'] = data['Low'].rolling(window=20).min()
        data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5
        data['Momentum'] = data['Close'].pct_change().rolling(window=10).sum()
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = compute_rsi(data['Close'])

        signal = ""
        trade_flag = False
        rank_value = 0

        try:
            close = data['Close'].iloc[-1]
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            vwap = data['VWAP'].iloc[-1] if 'VWAP' in data.columns else None
            open_ = data['Open'].iloc[-1]
            prev_close = data['Close'].iloc[-2]

            if strategy == "Breakout":
