import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import base64
import os
from collections import defaultdict
from streamlit_autorefresh import st_autorefresh

# --- AI Watchlist ---
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "PLTR", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "CRWV", "DDOG"
]

# --- Leaderboard History ---
signal_leaderboard = defaultdict(int)

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("\U0001F4C8 Day Trading Dashboard")
strategy = st.sidebar.selectbox("Select Strategy", ["Breakout", "Scalping", "Trend Trading"])
refresh_rate = st.sidebar.slider("Refresh every N seconds", 30, 300, 60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# Strategy Definitions – Always Visible
st.sidebar.markdown("### \U0001F4D8 Strategy Definitions")
st.sidebar.markdown("""
**Breakout**  
Triggered when price breaks above recent highs *and* trades above intraday VWAP — a sign of bullish conviction.

**Scalping**  
Short, fast trades triggered by volume surges and a 20MA crossing above 50MA.

**Trend Trading**  
Looks for steady momentum: 20MA > 50MA means upward trend likely continuing.
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
    f.write(b"ID3\x03\x00\x00\x00\x00\x00\x21TIT2\x00\x00\x00\x07\x00\x00\x03Beep\x00\x00")

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

        data.index = data.index.tz_localize(None)
        market_open = data.between_time("09:30", "16:00")
        data = market_open.copy()

        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['High_Break'] = data['High'].rolling(window=20).max()
        data['Low_Break'] = data['Low'].rolling(window=20).min()
        data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5
        data['Momentum'] = data['Close'].pct_change().rolling(window=10).sum()

        # VWAP Calculation
        if all(col in data.columns for col in ['High', 'Low', 'Close', 'Volume']):
            data['Typical_Price'] = (
                data['High'].fillna(0) + data['Low'].fillna(0) + data['Close'].fillna(0)
            ) / 3
            data['TPxV'] = data['Typical_Price'].fillna(0).astype(float) * data['Volume'].fillna(0).astype(float)
            data['VWAP'] = data['TPxV'].cumsum() / data['Volume'].fillna(0).cumsum()

        signal = ""
        trade_flag = False
        rank_value = 0

        try:
            if strategy == "Breakout":
                recent_high = data['High_Break'].iloc[-1].item()
                current_close = data['Close'].iloc[-1].item()
                current_vwap = data['VWAP'].iloc[-1].item()
                if pd.notna(recent_high) and pd.notna(current_close) and pd.notna(current_vwap) \
                   and current_close > recent_high and current_close > current_vwap:
                    signal = f"\U0001F514 Breakout: {ticker} above ${recent_high:.2f} & VWAP"
                    trade_flag = True
                    rank_value = data['Momentum'].iloc[-1].item()

            elif strategy == "Scalping":
                ma_20 = data['20_MA'].iloc[-1].item()
                ma_50 = data['50_MA'].iloc[-1].item()
                volume_surge = bool(data['Volume_Surge'].iloc[-1])
                if pd.notna(ma_20) and pd.notna(ma_50) and volume_surge and ma_20 > ma_50:
                    signal = f"⚡ Scalping: {ticker} volume surge & 20MA > 50MA"
                    trade_flag = True
                    rank_value = data['Volume'].iloc[-1].item()

            elif strategy == "Trend Trading":
                ma_20 = data['20_MA'].iloc[-1].item()
                ma_50 = data['50_MA'].iloc[-1].item()
                if pd.notna(ma_20) and pd.notna(ma_50) and ma_20 > ma_50:
                    signal = f"\U0001F4C8 Trend: {ticker} in uptrend (20MA > 50MA)"
                    trade_flag = True
                    rank_value = data['Momentum'].iloc[-1].item()
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")

        if trade_flag:
            ranked_signals.append((ticker, signal, rank_value))
            signal_leaderboard[ticker] += 1
            play_alert()

    if ranked_signals:
        st.markdown("### \U0001F4CA Real-Time Signals")
        ranked_signals.sort(key=lambda x: x[2], reverse=True)
        for ticker, signal, rank in ranked_signals:
            st.success(signal)

    if signal_leaderboard:
        leaderboard_df = pd.DataFrame(sorted(signal_leaderboard.items(), key=lambda x: x[1], reverse=True), columns=['Ticker', 'Signal Count'])
        st.markdown("### \U0001F3C6 Signal Leaderboard")
        st.dataframe(leaderboard_df)

# --- Optional: Download Script from App ---
try:
    with open(__file__, "r") as f:
        full_code = f.read()

    st.download_button(
        label="\U0001F4E5 Download Updated Script",
        data=full_code,
        file_name="trading_dashboard.py",
        mime="text/plain"
    )
except:
    st.warning("Script download not available in this environment.")
