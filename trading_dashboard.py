import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go
import base64
import time

# --- AI Watchlist ---
AI_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "PLTR", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "CRWV", "DDOG"
]

# --- Leaderboard History ---
from collections import defaultdict
signal_leaderboard = defaultdict(int)

# --- UI ---
st.set_page_config(layout="wide")
st.title("\U0001F4C8 Day Trading Dashboard")
strategy = st.sidebar.selectbox("Select Strategy", ["Breakout", "Scalping", "Trend Trading"])
use_ai_watchlist = st.sidebar.checkbox("Use AI Company Watchlist", value=False)
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (comma-separated)", value="AAPL")
refresh_rate = st.sidebar.slider("Refresh every N seconds", 30, 300, 60, step=10)

tickers = AI_TICKERS if use_ai_watchlist else [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

placeholder = st.empty()

while True:
    with placeholder.container():
        now = datetime.datetime.now()
        start_date = now - datetime.timedelta(days=5)
        end_date = now
        ranked_signals = []
        index_prices = []

        for ticker in tickers:
            st.subheader(f"Loading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, interval="5m")
            data.dropna(inplace=True)

            if data.empty or len(data) < 50:
                st.error(f"Not enough data for {ticker}. Try a different ticker or wait for market hours.")
                continue

            # Filter to regular market hours (9:30 AM to 4:00 PM)
            data = data.between_time("09:30", "16:00")

            # --- Zoom in on last 2 days ---
            data = data[data.index > (data.index[-1] - pd.Timedelta(days=2))]

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

            if use_ai_watchlist:
                index_prices.append(data['Close'].iloc[-1])

            # --- Display Signal ---
            if trade_flag:
                ranked_signals.append((ticker, signal, rank_value))
                signal_leaderboard[ticker] += 1

            # --- VWAP calculation ---
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

            # --- RSI calculation ---
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # --- Candlestick Chart ---
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'
            )])
            fig.add_trace(go.Scatter(x=data.index, y=data['20_MA'], mode='lines', name='20 MA'))
            fig.add_trace(go.Scatter(x=data.index, y=data['50_MA'], mode='lines', name='50 MA'))
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

            if trade_flag:
                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[data['Close'].iloc[-1]],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=['⬆ Signal'],
                    textposition='top center',
                    name='Trade Signal'
                ))

            fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig)

            # --- RSI Chart ---
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
            rsi_fig.update_layout(title=f"{ticker} RSI (14)", xaxis_title="Time", yaxis_title="RSI")
            rsi_fig.update_yaxes(range=[0, 100])
            st.plotly_chart(rsi_fig)

        # --- AI Sector Index ---
        if use_ai_watchlist and index_prices:
            avg_price = sum(index_prices) / len(index_prices)
            color = 'green' if avg_price >= index_prices[-2] else 'red'
            st.markdown(f"<h3 style='color:{color};'>AI Sector Index Avg Price: ${avg_price:.2f}</h3>", unsafe_allow_html=True)

        # --- Leaderboard ---
        if signal_leaderboard:
            leaderboard_df = pd.DataFrame(sorted(signal_leaderboard.items(), key=lambda x: x[1], reverse=True), columns=['Ticker', 'Signal Count'])
            st.markdown("### \U0001F3C6 AI Signal Leaderboard")
            st.dataframe(leaderboard_df)

    time.sleep(refresh_rate)
    placeholder.empty()
