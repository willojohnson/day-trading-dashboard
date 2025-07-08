import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go
import base64
import time

# --- UI ---
st.set_page_config(layout="wide")
st.title("\U0001F4C8 Day Trading Dashboard")
strategy = st.sidebar.selectbox("Select Strategy", ["Breakout", "Scalping", "Trend Trading"])
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (comma-separated)", value="AAPL")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
refresh_rate = st.sidebar.slider("Refresh every N seconds", 30, 300, 60, step=10)

placeholder = st.empty()

while True:
    with placeholder.container():
        now = datetime.datetime.now()
        start_date = now - datetime.timedelta(days=5)
        end_date = now
        ranked_signals = []

        for ticker in tickers:
            st.subheader(f"Loading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, interval="5m")
            data.dropna(inplace=True)

            if data.empty or len(data) < 50:
                st.error(f"Not enough data for {ticker}. Try a different ticker or wait for market hours.")
                continue

            # --- Indicators ---
            data['20_MA'] = data['Close'].rolling(window=20).mean()
            data['50_MA'] = data['Close'].rolling(window=50).mean()
            data['High_Break'] = data['High'].rolling(window=20).max()
            data['Low_Break'] = data['Low'].rolling(window=20).min()
            data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5
            data['Momentum'] = data['Close'].pct_change().rolling(window=10).sum()

            # --- Signal Logic ---
            signal = ""
            trade_flag = False
            rank_value = 0

            try:
                if strategy == "Breakout":
                    recent_high = data['High_Break'].iloc[-1].item()
                    current_close = data['Close'].iloc[-1].item()
                    if pd.notna(recent_high) and pd.notna(current_close) and current_close > recent_high:
                        signal = (f"\U0001F514 Breakout Alert: {ticker} is breaking above recent resistance at ${recent_high:.2f}")
                        trade_flag = True
                        rank_value = data['Momentum'].iloc[-1].item()

                elif strategy == "Scalping":
                    ma_20 = data['20_MA'].iloc[-1].item()
                    ma_50 = data['50_MA'].iloc[-1].item()
                    volume_surge = bool(data['Volume_Surge'].iloc[-1].item()) if hasattr(data['Volume_Surge'].iloc[-1], 'item') else bool(data['Volume_Surge'].iloc[-1])
                    if pd.notna(ma_20) and pd.notna(ma_50) and volume_surge and ma_20 > ma_50:
                        signal = f"⚡ Scalping Opportunity: {ticker} has volume surge + short-term MA crossover"
                        trade_flag = True
                        rank_value = data['Volume'].iloc[-1].item()

                elif strategy == "Trend Trading":
                    ma_20 = data['20_MA'].iloc[-1].item()
                    ma_50 = data['50_MA'].iloc[-1].item()
                    if pd.notna(ma_20) and pd.notna(ma_50) and ma_20 > ma_50:
                        signal = f"\U0001F4C8 Trend Trade: {ticker} in uptrend (20 MA > 50 MA). Consider riding the wave."
                        trade_flag = True
                        rank_value = data['Momentum'].iloc[-1].item()

            except Exception as e:
                st.warning(f"Signal calculation error for {ticker}: {e}")

            # --- Display Signal ---
            if trade_flag:
                ranked_signals.append((ticker, signal, rank_value))

        # --- Rank and Display Top Trade Signals ---
        if ranked_signals:
            sorted_signals = sorted(ranked_signals, key=lambda x: x[2], reverse=True)
            for ticker, signal, rank in sorted_signals:
                st.markdown(f"<h2 style='color:limegreen;'>✅ TRADE SIGNAL: {ticker}</h2>", unsafe_allow_html=True)
                st.success(signal)
                # Sound Alert: plays only once per trade signal batch
                sound_file = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
                st.markdown(f"<audio autoplay><source src='{sound_file}' type='audio/wav'></audio>", unsafe_allow_html=True)

        else:
            st.info("No clear signals found for any ticker.")

    time.sleep(refresh_rate)
    placeholder.empty()
