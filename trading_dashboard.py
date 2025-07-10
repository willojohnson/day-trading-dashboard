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

# --- Leaderboard History (session state for persistence) ---
if 'signal_leaderboard' not in st.session_state:
    st.session_state.signal_leaderboard = defaultdict(int)
signal_leaderboard = st.session_state.signal_leaderboard

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

# --- Sound alert function ---
def play_alert():
    sound_file_path = "alert.mp3"
    if os.path.exists(sound_file_path):
        with open(sound_file_path, "rb") as f:
            b64_sound = base64.b64encode(f.read()).decode()
        sound_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_sound}" type="audio/mp3">
            </audio>"""
        st.markdown(sound_html, unsafe_allow_html=True)

# --- Only create dummy alert file if it doesn't exist ---
if not os.path.exists("alert.mp3"):
    with open("alert.mp3", "wb") as f:
        f.write(b"ID3\x03\x00\x00\x00\x00\x00\x21TIT2\x00\x00\x00\x07\x00\x00\x03Beep\x00\x00")

# --- Helper function for safe scalar extraction ---
def get_scalar(val):
    return val.item() if hasattr(val, "item") else val

placeholder = st.empty()

with placeholder.container():
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=5)
    end_date = now
    ranked_signals = []

    for ticker in TICKERS:
        st.subheader(f"Loading data for {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {e}")
            continue

        if data.empty or len(data) < 50 or 'Close' not in data.columns:
            st.error(f"Not enough or invalid data for {ticker}.")
            continue

        # Remove timezone if present
        try:
            if hasattr(data.index, "tz") and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
        except Exception:
            pass

        # Filter for regular market hours and copy
        try:
            data = data.between_time("09:30", "16:00").copy()
        except Exception:
            pass

        # If after filtering, data is empty, skip
        if data.empty or len(data) < 50 or 'Close' not in data.columns:
            st.error(f"Not enough or invalid data for {ticker} after market hours filter.")
            continue

        # All calculations and assignments must use the filtered DataFrame only!
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['High_Break'] = data['High'].rolling(window=20).max()
        data['Low_Break'] = data['Low'].rolling(window=20).min()
        data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5
        data['Momentum'] = data['Close'].pct_change().rolling(window=10).sum()

        # --- VWAP Calculation: robust against length mismatch ---
        if all(col in data.columns for col in ['High', 'Low', 'Close', 'Volume']):
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['TPxV'] = (data['Typical_Price'] * data['Volume']).reindex(data.index)
        vwap_numerator = data['TPxV'].cumsum().reindex(data.index)
        vwap_denominator = data['Volume'].cumsum().replace(0, 1e-9).reindex(data.index)
        data['VWAP'] = (vwap_numerator / vwap_denominator).reindex(data.index)

        signal = ""
        trade_flag = False
        rank_value = 0

        try:
            if strategy == "Breakout":
                recent_high = get_scalar(data['High_Break'].iloc[-1])
                current_close = get_scalar(data['Close'].iloc[-1])
                current_vwap = get_scalar(data['VWAP'].iloc[-1])
                if pd.notna(recent_high) and pd.notna(current_close) and pd.notna(current_vwap) \
                   and current_close > recent_high and current_close > current_vwap:
                    signal = f"\U0001F514 Breakout: {ticker} above ${recent_high:.2f} & VWAP"
                    trade_flag = True
                    rank_value = get_scalar(data['Momentum'].iloc[-1])

            elif strategy == "Scalping":
                ma_20 = get_scalar(data['20_MA'].iloc[-1])
                ma_50 = get_scalar(data['50_MA'].iloc[-1])
                volume_surge = bool(data['Volume_Surge'].iloc[-1])
                if pd.notna(ma_20) and pd.notna(ma_50) and volume_surge and ma_20 > ma_50:
                    signal = f"⚡ Scalping: {ticker} volume surge & 20MA > 50MA"
                    trade_flag = True
                    rank_value = get_scalar(data['Volume'].iloc[-1])

            elif strategy == "Trend Trading":
                ma_20 = get_scalar(data['20_MA'].iloc[-1])
                ma_50 = get_scalar(data['50_MA'].iloc[-1])
                if pd.notna(ma_20) and pd.notna(ma_50) and ma_20 > ma_50:
                    signal = f"\U0001F4C8 Trend: {ticker} in uptrend (20MA > 50MA)"
                    trade_flag = True
                    rank_value = get_scalar(data['Momentum'].iloc[-1])
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
        import sys
        script_path = sys.argv[0]
        with open(script_path, "r", encoding="utf-8") as f:
            full_code = f.read()
        st.download_button(
            label="\U0001F4E5 Download Updated Script",
            data=full_code,
            file_name="trading_dashboard.py",
            mime="text/plain"
        )
    except Exception:
        st.info("Script download not available in this environment.")
