import streamlit as st
import yfinance as yf
import pandas as pd
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
