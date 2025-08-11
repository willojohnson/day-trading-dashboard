import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --- Tickers to Monitor ---
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI",
    "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG", "BTSG"
]

# --- Ticker to Company Name Mapping ---
TICKER_NAMES = {
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "SNOW": "Snowflake Inc.",
    "AI": "C3.ai Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "BBAI": "BigBear.ai Holdings Inc.",
    "SOUN": "SoundHound AI Inc.",
    "CRSP": "CRISPR Therapeutics AG",
    "TSM": "Taiwan Semiconductor",
    "DDOG": "Datadog Inc.",
    "BTSG": "BrightSpring Health Services"
}

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard")

# --- Tabs for navigation ---
tab1, tab2 = st.tabs(["Dashboard Overview", "Chart Analysis"]) 

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=10, max_value=300, value=30, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Interactive Strategy Selectors ---
all_bullish_strategies = [
    "Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Golden Cross",
    "Trend + MACD Bullish", "Ichimoku Bullish"
]
all_bearish_strategies = [
    "MACD Bearish Crossover", "RSI Overbought", "Death Cross",
    "Death Cross + RSI Bearish", "Ichimoku Bearish"
]
all_strategies = all_bullish_strategies + all_bearish_strategies

st.sidebar.markdown("### 🚦 Select Strategies")
selected_bullish = st.sidebar.multiselect("Bullish Strategies", all_bullish_strategies, default=all_bullish_strategies)
selected_bearish = st.sidebar.multiselect("Bearish Strategies", all_bearish_strategies, default=all_bearish_strategies)
selected_strategies = selected_bullish + selected_bearish

# --- Timeframe Selector ---
timeframe_options = ["5m", "15m", "30m", "1h", "1d", "3 month", "6 month", "YTD", "1 year", "5 year"]
timeframe = st.sidebar.selectbox("Select Timeframe", timeframe_options, index=4)

# --- Collapsible Strategy Definitions Section ---
with st.sidebar.expander("📘 Strategy Definitions", expanded=False):
    st.markdown("**Trend Trading**: 20MA > 50MA (trigger = cross from ≤ to >)")
    st.markdown("**RSI Overbought**: RSI > 70 (trigger = cross above 70)")
    st.markdown("**RSI Oversold**: RSI < 30 (trigger = cross below 30)")
    st.markdown("**MACD Bullish Crossover**: MACD crosses above Signal")
    st.markdown("**MACD Bearish Crossover**: MACD crosses below Signal")
    st.markdown("**Death Cross**: 50MA crosses below 200MA")
    st.markdown("**Golden Cross**: 50MA crosses above 200MA")
    st.markdown("---")
    st.markdown("### Ichimoku Cloud")
    st.markdown("- **Bullish**: Price crosses from below the cloud to above the cloud")
    st.markdown("- **Bearish**: Price crosses from above the cloud to below the cloud")
    st.markdown("---")
    st.markdown("**Trend + MACD Bullish**: Trend cross up AND MACD cross up on the same bar")
    st.markdown("**Death Cross + RSI Bearish**: Death Cross bar with RSI > 70 on the same bar")


# --- Helper: fetch and process data ---
@st.cache_data(ttl=refresh_rate)
def fetch_and_process_data(ticker: str, timeframe: str):
    end_date = datetime.datetime.now()
    start_date = None
    interval = "1d"

    if timeframe in ["5m", "15m", "30m", "1h"]:
        start_date = end_date - datetime.timedelta(days=7)
        interval = timeframe
    elif timeframe == "1d":
        start_date = end_date - datetime.timedelta(days=365)
        interval = "1d"
    elif timeframe == "3 month":
        start_date = end_date - datetime.timedelta(days=90)
    elif timeframe == "6 month":
        start_date = end_date - datetime.timedelta(days=180)
    elif timeframe == "YTD":
        start_date = datetime.datetime(end_date.year, 1, 1)
    elif timeframe == "1 year":
        start_date = end_date - datetime.timedelta(days=365)
    elif timeframe == "5 year":
        start_date = end_date - datetime.timedelta(days=5 * 365)

    if start_date is None:
        return None, "Unsupported timeframe."

    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty or "Close" not in df.columns:
        return None, "No valid data."

    # Indicators
    df["20_MA"] = df["Close"].rolling(window=20).mean()
    df["50_MA"] = df["Close"].rolling(window=50).mean()
    df["200_MA"] = df["Close"].rolling(window=200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Ichimoku
    high_9 = df["High"].rolling(window=9).max()
    low_9 = df["Low"].rolling(window=9).min()
    df["tenkan_sen"] = (high_9 + low_9) / 2

    high_26 = df["High"].rolling(window=26).max()
    low_26 = df["Low"].rolling(window=26).min()
    df["kijun_sen"] = (high_26 + low_26) / 2

    df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)

    high_52 = df["High"].rolling(window=52).max()
    low_52 = df["Low"].rolling(window=52).min()
    df["senkou_span_b"] = ((high_52 + low_52) / 2).shift(26)

    df["chikou_span"] = df["Close"].shift(-26)

    return df, None

# --- Helpers to compute signal triggers and returns ---
def last_cross_index(series_a: pd.Series, series_b: pd.Series, direction: str):
    """Find last index where a cross happened.
    direction: 'up' for a crossing from below to above, 'down' for above to below
    Returns the integer index or None.
    """
    a = series_a.values
    b = series_b.values
    if len(a) < 2:
        return None
    if direction == "up":
        cross = (a[:-1] <= b[:-1]) & (a[1:] > b[1:])
    else:
        cross = (a[:-1] >= b[:-1]) & (a[1:] < b[1:])
    idxs = np.where(cross)[0]
    return int(idxs[-1] + 1) if idxs.size else None


def last_threshold_cross_index(series: pd.Series, thresh: float, direction: str):
    """Cross relative to a threshold (e.g., RSI 30/70). direction 'up' or 'down'."""
    s = series.values
    if len(s) < 2:
        return None
    if direction == "down":
        cross = (s[:-1] >= thresh) & (s[1:] < thresh)
    else:
        cross = (s[:-1] <= thresh) & (s[1:] > thresh)
    idxs = np.where(cross)[0]
    return int(idxs[-1] + 1) if idxs.size else None


def ichimoku_cross_index(df: pd.DataFrame, direction: str):
    """Price crossing the cloud. direction 'up' or 'down'."""
    close = df["Close"].values
    a = df["senkou_span_a"].values
    b = df["senkou_span_b"].values
    if len(close) < 2:
        return None
    cloud_prev = np.maximum(a[:-1], b[:-1])
    cloud_now = np.maximum(a[1:], b[1:])
    floor_prev = np.minimum(a[:-1], b[:-1])
    floor_now = np.minimum(a[1:], b[1:])

    if direction == "up":
        cross = (close[:-1] <= cloud_prev) & (close[1:] > cloud_now)
    else:
        cross = (close[:-1] >= floor_prev) & (close[1:] < floor_now)
    idxs = np.where(cross)[0]
    return int(idxs[-1] + 1) if idxs.size else None


def compute_return_since(df: pd.DataFrame, trigger_idx: int, side: str) -> float:
    """Return % from trigger close to latest close. side: 'long' or 'short'."""
    if trigger_idx is None or trigger_idx >= len(df):
        return 0.0
    entry = float(df["Close"].iloc[trigger_idx])
    last = float(df["Close"].iloc[-1])
    if side == "long":
        r = (last / entry - 1.0) * 100.0
    else:
        r = (entry / last - 1.0) * 100.0
    return float(r)


# --- Main Logic ---
signals = []
# Grid holds returns. Positive = bullish (green), Negative = bearish (red), 0 = no signal (light gray)
heatmap_data = {strategy: np.zeros(len(TICKERS), dtype=float) for strategy in all_strategies}
heatmap_hover_text = {strategy: [""] * len(TICKERS) for strategy in all_strategies}

with st.spinner("⚙️ Processing data and generating signals..."):
    if selected_strategies:
        for i, ticker in enumerate(TICKERS):
            company = TICKER_NAMES.get(ticker, ticker)
            df, error_msg = fetch_and_process_data(ticker, timeframe)

            if error_msg:
                st.warning(f"⚠️ {error_msg} for {ticker} ({company}). Using neutral values.")
                continue
            if len(df) < 200:
                st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators. Some strategies may not trigger.")

            # --- Compute last triggers and returns per strategy ---
            # Bullish
            if "Trend Trading" in selected_bullish:
                idx = last_cross_index(df["20_MA"], df["50_MA"], "up")
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["Trend Trading"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company} | since {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Trend Trading"][i] = f"20>50 cross on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            if "RSI Oversold" in selected_bullish:
                idx = last_threshold_cross_index(df["RSI"], 30.0, "down")
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["RSI Oversold"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} | {ret:.2f}%"))
                    heatmap_hover_text["RSI Oversold"][i] = f"Cross <30 on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            if "MACD Bullish Crossover" in selected_bullish:
                idx = last_cross_index(df["MACD"], df["MACD_Signal"], "up")
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["MACD Bullish Crossover"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"📈 Bullish - MACD Cross Up — {company} | {ret:.2f}%"))
                    heatmap_hover_text["MACD Bullish Crossover"][i] = f"Cross up on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            if "Golden Cross" in selected_bullish:
                idx = last_cross_index(df["50_MA"], df["200_MA"], "up")
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["Golden Cross"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Golden Cross"][i] = f"50>200 on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            if "Trend + MACD Bullish" in selected_bullish:
                idx_trend = last_cross_index(df["20_MA"], df["50_MA"], "up")
                idx_macd = last_cross_index(df["MACD"], df["MACD_Signal"], "up")
                idx = idx_trend if (idx_trend is not None and idx_trend == idx_macd) else None
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["Trend + MACD Bullish"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"✨ Bullish - Trend+MACD — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Trend + MACD Bullish"][i] = f"Both on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            if "Ichimoku Bullish" in selected_bullish:
                idx = ichimoku_cross_index(df, "up")
                ret = compute_return_since(df, idx, "long") if idx is not None else 0.0
                heatmap_data["Ichimoku Bullish"][i] = ret
                if idx is not None:
                    signals.append((ticker, "bullish", f"☁️ Bullish - Ichimoku Breakout — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Ichimoku Bullish"][i] = f"Above cloud on {df.index[idx].date()}<br>Return: {ret:.2f}%"

            # Bearish (store as negative so they render red)
            if "RSI Overbought" in selected_bearish:
                idx = last_threshold_cross_index(df["RSI"], 70.0, "up")
                ret = compute_return_since(df, idx, "short") if idx is not None else 0.0
                heatmap_data["RSI Overbought"][i] = -ret
                if idx is not None:
                    signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} | {ret:.2f}%"))
                    heatmap_hover_text["RSI Overbought"][i] = f"Cross >70 on {df.index[idx].date()}<br>Short Return: {ret:.2f}%"

            if "MACD Bearish Crossover" in selected_bearish:
                idx = last_cross_index(df["MACD"], df["MACD_Signal"], "down")
                ret = compute_return_since(df, idx, "short") if idx is not None else 0.0
                heatmap_data["MACD Bearish Crossover"][i] = -ret
                if idx is not None:
                    signals.append((ticker, "bearish", f"📉 Bearish - MACD Cross Down — {company} | {ret:.2f}%"))
                    heatmap_hover_text["MACD Bearish Crossover"][i] = f"Cross down on {df.index[idx].date()}<br>Short Return: {ret:.2f}%"

            if "Death Cross" in selected_bearish:
                idx = last_cross_index(df["50_MA"], df["200_MA"], "down")
                ret = compute_return_since(df, idx, "short") if idx is not None else 0.0
                heatmap_data["Death Cross"][i] = -ret
                if idx is not None:
                    signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Death Cross"][i] = f"50<200 on {df.index[idx].date()}<br>Short Return: {ret:.2f}%"

            if "Death Cross + RSI Bearish" in selected_bearish:
                idx_dc = last_cross_index(df["50_MA"], df["200_MA"], "down")
                idx_rsi = last_threshold_cross_index(df["RSI"], 70.0, "up")
                idx = idx_dc if (idx_dc is not None and idx_dc == idx_rsi) else None
                ret = compute_return_since(df, idx, "short") if idx is not None else 0.0
                heatmap_data["Death Cross + RSI Bearish"][i] = -ret
                if idx is not None:
                    signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Death Cross + RSI Bearish"][i] = f"Both on {df.index[idx].date()}<br>Short Return: {ret:.2f}%"

            if "Ichimoku Bearish" in selected_bearish:
                idx = ichimoku_cross_index(df, "down")
                ret = compute_return_since(df, idx, "short") if idx is not None else 0.0
                heatmap_data["Ichimoku Bearish"][i] = -ret
                if idx is not None:
                    signals.append((ticker, "bearish", f"☁️ Bearish - Ichimoku Breakdown — {company} | {ret:.2f}%"))
                    heatmap_hover_text["Ichimoku Bearish"][i] = f"Below cloud on {df.index[idx].date()}<br>Short Return: {ret:.2f}%"
    else:
        st.info("⚠️ Please select at least one strategy from the sidebar to generate signals.")

# Create DataFrames for the heatmap
heatmap_df = pd.DataFrame(heatmap_data, index=TICKERS)
heatmap_hover_df = pd.DataFrame(heatmap_hover_text, index=TICKERS)

# --- DASHBOARD OVERVIEW TAB ---
with tab1:
    # --- KPI Metrics at the Top ---
    st.markdown("### 📊 Market Overview")
    kpi_ticker = st.selectbox("Select a Ticker to view KPIs", TICKERS, index=0, key="kpi_select")

    kpi_df, _ = fetch_and_process_data(kpi_ticker, "1 year")

    if kpi_df is not None and not kpi_df.empty and len(kpi_df) >= 2:
        latest_price = float(kpi_df["Close"].iloc[-1])
        previous_price = float(kpi_df["Close"].iloc[-2])
        current_volume = int(kpi_df["Volume"].iloc[-1])
        low_price = float(kpi_df["Low"].iloc[-1])
        high_price = float(kpi_df["High"].iloc[-1])
        avg_volume = float(kpi_df["Volume"].mean())

        change_pct = ((latest_price - previous_price) / previous_price) * 100 if previous_price else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=f"Price ({kpi_ticker})", value=f"${latest_price:.2f}", delta=f"{change_pct:.2f}%")
        with col2:
            st.metric(label="Today's Range", value=f"${low_price:.2f} - ${high_price:.2f}")
        with col3:
            st.metric(label="Current Volume", value=f"{current_volume:,}")
        with col4:
            st.metric(label="Average Volume", value=f"{avg_volume:,.0f}")
    else:
        st.warning(f"⚠️ Insufficient data for {kpi_ticker} to calculate KPIs. Please check back later.")

    st.markdown("---")

    # --- Signal Display ---
    st.markdown("### ✅ Current Trade Signals")
    if signals:
        for _, signal_type, msg in signals:
            if signal_type == "bullish":
                st.success(msg)
            else:
                st.error(msg)
    elif selected_strategies:
        st.info("No trade signals at this time for any active strategies.")
    else:
        st.info("Please select strategies from the sidebar to see current trade signals.")

    st.markdown("---")

    # --- Strategy Heatmap ---
    st.markdown("### 📊 Strategy Heatmap")
    if not heatmap_df.empty:
        # Determine symmetric range for diverging colors
        max_abs = np.nanmax(np.abs(heatmap_df.values)) if heatmap_df.size else 1.0
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0

        # Diverging colorscale: red -> light gray -> green
        colorscale = [
            [0.0, "rgb(180,0,0)"],
            [0.5, "rgb(220,220,220)"],
            [1.0, "rgb(0,180,0)"]
        ]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=list(heatmap_df.columns),
            y=list(heatmap_df.index),
            colorscale=colorscale,
            zmin=-max_abs,
            zmax=+max_abs,
            zmid=0,
            colorbar=dict(title="Return %", ticksuffix="%"),
            text=heatmap_hover_df.values,
            hoverinfo='text'
        ))

        fig_heatmap.update_layout(
            title="Active Signals by Strategy and Ticker",
            xaxis_title="Strategies",
            yaxis_title="Tickers",
            xaxis=dict(tickangle=45, type='category', constrain='domain'),
            yaxis=dict(autorange='reversed', type='category', constrain='domain'),
            autosize=True,
            margin=dict(l=60, r=10, t=60, b=120)
        )
        fig_heatmap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
        fig_heatmap.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')

        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

# --- CHART ANALYSIS TAB ---
with tab2:
    st.markdown("### 🔎 Detailed Chart Analysis")
    chart_ticker = st.selectbox("Select a Ticker", options=TICKERS, key="chart_ticker")

    if chart_ticker:
        chart_df, error_msg = fetch_and_process_data(chart_ticker, timeframe)

        if chart_df is not None and not chart_df.empty:
            fig_candlestick = go.Figure(data=[
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['Open'],
                    high=chart_df['High'],
                    low=chart_df['Low'],
                    close=chart_df['Close'],
                    name='Price'
                ),
                go.Scatter(x=chart_df.index, y=chart_df['20_MA'], mode='lines', name='20 MA', line=dict(color='orange', width=2)),
                go.Scatter(x=chart_df.index, y=chart_df['50_MA'], mode='lines', name='50 MA', line=dict(color='blue', width=2)),
                go.Scatter(x=chart_df.index, y=chart_df['200_MA'], mode='lines', name='200 MA', line=dict(color='red', width=2))
            ])

            fig_candlestick.update_layout(
                title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Price Action (Moving Averages)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_candlestick, use_container_width=True)

            fig_ichimoku = go.Figure(data=[
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['Open'],
                    high=chart_df['High'],
                    low=chart_df['Low'],
                    close=chart_df['Close'],
                    name='Price'
                ),
                go.Scatter(x=chart_df.index, y=chart_df['tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='red', width=1)),
                go.Scatter(x=chart_df.index, y=chart_df['kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='blue', width=1)),
                go.Scatter(x=chart_df.index, y=chart_df['chikou_span'], mode='lines', name='Chikou Span', line=dict(color='green', width=1)),
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df['senkou_span_a'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Cloud'
                ),
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df['senkou_span_b'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    fillcolor='rgba(255, 0, 0, 0.2)' if chart_df['senkou_span_a'].iloc[-1] > chart_df['senkou_span_b'].iloc[-1] else 'rgba(0, 255, 0, 0.2)',
                    name='Cloud Fill'
                )
            ])

            fig_ichimoku.update_layout(
                title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Ichimoku Cloud ({timeframe} interval)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_ichimoku, use_container_width=True)
        else:
            st.warning(f"⚠️ No data available for {chart_ticker} at the selected timeframe.")
