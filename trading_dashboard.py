import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# =============================================================
# Trading Dashboard – Hotfix v2
# - Fix ambiguous truth on valid.sum() by using numpy array
# - Ensure ALL cross-index helpers return **positional** indices for .iloc
# - Keep all prior rewired improvements (RSI re-entry, slope, combo window, ATR filter, etc.)
# =============================================================

# --- Tickers to Monitor ---
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI",
    "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG", "BTSG"
]

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

st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard (Hotfix v2)")

tab1, tab2 = st.tabs(["Dashboard Overview", "Chart Analysis"]) 

st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=10, max_value=300, value=30, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

all_bullish_strategies = [
    "Trend Trading", "MACD Bullish Crossover", "RSI Oversold (Re-entry)", "Golden Cross",
    "Trend + MACD Bullish", "Ichimoku Bullish"
]
all_bearish_strategies = [
    "MACD Bearish Crossover", "RSI Overbought (Re-entry)", "Death Cross",
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

# --- Signal Filters / Options ---
st.sidebar.markdown("### 🔧 Signal Filters & Options")
MA_SLOPE_FILTER = st.sidebar.checkbox("Require MA slope confirmation (20/50/200)", value=True)
COMBO_WINDOW = st.sidebar.number_input("Combo window (bars)", min_value=0, max_value=10, value=3, step=1)
MACD_ATR_FILTER_ON = st.sidebar.checkbox("Require minimum MACD delta vs Signal (ATR-based)", value=True)
MACD_ATR_MULT = st.sidebar.slider("MACD delta ≥ ATR × multiplier", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
ICHIMOKU_REQUIRE_COLOR = st.sidebar.checkbox("Ichimoku: require bullish/bearish cloud color", value=True)
ICHIMOKU_REQUIRE_CHIKOU = st.sidebar.checkbox("Ichimoku: require Chikou confirmation", value=True)
COLOR_CLAMP = st.sidebar.slider("Heatmap color clamp (abs %)", 10, 100, 50, 5)

with st.sidebar.expander("📘 Strategy Definitions", expanded=False):
    st.markdown("**Trend Trading**: 20MA cross **up** above 50MA (with optional slope filter)")
    st.markdown("**RSI Oversold (Re-entry)**: RSI crosses **up** back above 30")
    st.markdown("**RSI Overbought (Re-entry)**: RSI crosses **down** back below 70")
    st.markdown("**MACD Bullish/Bearish**: MACD crosses above/below Signal; optional ATR-based min delta")
    st.markdown("**Golden Cross**: 50MA crosses **up** above 200MA (with optional slope filter)")
    st.markdown("**Death Cross**: 50MA crosses **down** below 200MA (with optional slope filter)")
    st.markdown("**Trend + MACD Bullish**: Trend cross up and MACD cross up within window")
    st.markdown("**Death Cross + RSI Bearish**: Death Cross and RSI re-entry below 70 within window")
    st.markdown("**Ichimoku Bullish/Bearish**: Close crosses cloud up/down; optional cloud color & Chikou confirmation")

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

    df = df.copy()

    # MAs
    df["20_MA"] = df["Close"].rolling(window=20).mean()
    df["50_MA"] = df["Close"].rolling(window=50).mean()
    df["200_MA"] = df["Close"].rolling(window=200).mean()

    # RSI (14)
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
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ATR (14)
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Ichimoku
    high_9 = df["High"].rolling(window=9).max(); low_9 = df["Low"].rolling(window=9).min()
    df["tenkan_sen"] = (high_9 + low_9) / 2
    high_26 = df["High"].rolling(window=26).max(); low_26 = df["Low"].rolling(window=26).min()
    df["kijun_sen"] = (high_26 + low_26) / 2
    df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
    high_52 = df["High"].rolling(window=52).max(); low_52 = df["Low"].rolling(window=52).min()
    df["senkou_span_b"] = ((high_52 + low_52) / 2).shift(26)
    df["chikou_span"] = df["Close"].shift(-26)

    return df, None

# ---------- Helpers (positional indices) ----------

def _nanmask(a: pd.Series, b: pd.Series):
    m = (~a.isna()) & (~b.isna())
    return a[m].values, b[m].values, np.where(m)[0]  # positional map


def last_cross_index(series_a: pd.Series, series_b: pd.Series, direction: str):
    a, b, posmap = _nanmask(series_a, series_b)
    if len(a) < 2:
        return None
    cross = (a[:-1] <= b[:-1]) & (a[1:] > b[1:]) if direction == 'up' else (a[:-1] >= b[:-1]) & (a[1:] < b[1:])
    idxs = np.where(cross)[0]
    return int(posmap[idxs[-1] + 1]) if idxs.size else None


def last_threshold_cross_index(series: pd.Series, thresh: float, mode: str):
    """mode ∈ {'breach_down','breach_up','reenter_above','reenter_below'} — returns **positional** index"""
    mask = ~series.isna().values
    s = series.values[mask]
    if len(s) < 2:
        return None
    posmap = np.where(mask)[0]
    if mode == 'breach_down':
        cross = (s[:-1] >= thresh) & (s[1:] < thresh)
    elif mode == 'breach_up':
        cross = (s[:-1] <= thresh) & (s[1:] > thresh)
    elif mode == 'reenter_above':
        cross = (s[:-1] < thresh) & (s[1:] >= thresh)
    else:  # reenter_below
        cross = (s[:-1] > thresh) & (s[1:] <= thresh)
    idxs = np.where(cross)[0]
    return int(posmap[idxs[-1] + 1]) if idxs.size else None


def ichimoku_cross_index(df: pd.DataFrame, direction: str, require_color: bool, require_chikou: bool):
    close = df['Close']; a = df['senkou_span_a']; b = df['senkou_span_b']
    valid = (~close.isna()) & (~a.isna()) & (~b.isna())
    valid_np = valid.to_numpy() if hasattr(valid, 'to_numpy') else np.asarray(valid)
    if valid_np.sum() < 2:
        return None
    c = close.values[valid_np]
    A = a.values[valid_np]
    B = b.values[valid_np]
    posmap = np.where(valid_np)[0]

    cloud_top_prev = np.maximum(A[:-1], B[:-1])
    cloud_top_now  = np.maximum(A[1:],  B[1:])
    cloud_bot_prev = np.minimum(A[:-1], B[:-1])
    cloud_bot_now  = np.minimum(A[1:],  B[1:])

    cross = (c[:-1] <= cloud_top_prev) & (c[1:] > cloud_top_now) if direction == 'up' \
            else (c[:-1] >= cloud_bot_prev) & (c[1:] < cloud_bot_now)

    idxs = np.where(cross)[0]
    if not idxs.size:
        return None

    for k in idxs[::-1]:
        pos = int(posmap[k + 1])
        ok = True
        if require_color:
            if direction == 'up' and not (df['senkou_span_a'].iloc[pos] > df['senkou_span_b'].iloc[pos]):
                ok = False
            if direction == 'down' and not (df['senkou_span_a'].iloc[pos] < df['senkou_span_b'].iloc[pos]):
                ok = False
        if ok and require_chikou and pos - 26 >= 0:
            price_back = df['Close'].iloc[pos - 26]
            chikou_now = df['chikou_span'].iloc[pos - 26]
            if direction == 'up' and not (chikou_now > price_back):
                ok = False
            if direction == 'down' and not (chikou_now < price_back):
                ok = False
        if ok:
            return pos
    return None


def compute_return_since(df: pd.DataFrame, trigger_idx: int, side: str) -> float:
    if trigger_idx is None or trigger_idx >= len(df):
        return 0.0
    entry = float(df['Close'].iloc[trigger_idx]); last = float(df['Close'].iloc[-1])
    if entry <= 0 or last <= 0:
        return 0.0
    return float(((last/entry - 1.0) if side == 'long' else (entry/last - 1.0)) * 100.0)


def slope(series: pd.Series, lookback: int = 5):
    if len(series.dropna()) <= lookback:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-lookback-1])

# ---------- Main Logic ----------
signals = []
heatmap_data = {s: np.zeros(len(TICKERS), dtype=float) for s in all_strategies}
heatmap_hover_text = {s: [""] * len(TICKERS) for s in all_strategies}

with st.spinner("⚙️ Processing data and generating signals..."):
    if selected_strategies:
        for i, ticker in enumerate(TICKERS):
            company = TICKER_NAMES.get(ticker, ticker)
            df, error_msg = fetch_and_process_data(ticker, timeframe)
            if error_msg:
                st.warning(f"⚠️ {error_msg} for {ticker} ({company}). Using neutral values.")
                continue
            if len(df) < 210:
                st.info(f"ℹ️ Limited data for {ticker}. Some strategies may not trigger.")

            s20 = slope(df['20_MA']); s50 = slope(df['50_MA']); s200 = slope(df['200_MA'])
            macd_ok = True
            if MACD_ATR_FILTER_ON:
                macd_ok = df['ATR'].iloc[-1] <= 0 or abs(df['MACD_Hist'].iloc[-1]) >= (MACD_ATR_MULT * df['ATR'].iloc[-1])

            # Bullish
            if "Trend Trading" in selected_bullish:
                idx_trend = last_cross_index(df["20_MA"], df["50_MA"], "up")
                if idx_trend is not None and (not MA_SLOPE_FILTER or (s20 > 0 and s50 >= 0)):
                    ret = compute_return_since(df, idx_trend, "long")
                    heatmap_data["Trend Trading"][i] = ret
                    signals.append((ticker, "bullish", f"📈 Trend Up — {company} | {df.index[idx_trend].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Trend Trading"][i] = f"20>50 on {df.index[idx_trend].date()}<br>Ret: {ret:.2f}%"

            if "RSI Oversold (Re-entry)" in selected_bullish:
                idx_rsi = last_threshold_cross_index(df["RSI"], 30.0, 'reenter_above')
                if idx_rsi is not None:
                    ret = compute_return_since(df, idx_rsi, "long")
                    heatmap_data["RSI Oversold (Re-entry)"][i] = ret
                    signals.append((ticker, "bullish", f"📈 RSI Re-entry >30 — {company} | {df.index[idx_rsi].date()} → {ret:.2f}%"))
                    heatmap_hover_text["RSI Oversold (Re-entry)"][i] = f"RSI↑ >30 on {df.index[idx_rsi].date()}<br>Ret: {ret:.2f}%"

            if "MACD Bullish Crossover" in selected_bullish:
                idx_macd_up = last_cross_index(df["MACD"], df["MACD_Signal"], "up")
                if idx_macd_up is not None and macd_ok:
                    ret = compute_return_since(df, idx_macd_up, "long")
                    heatmap_data["MACD Bullish Crossover"][i] = ret
                    signals.append((ticker, "bullish", f"📈 MACD Cross Up — {company} | {df.index[idx_macd_up].date()} → {ret:.2f}%"))
                    heatmap_hover_text["MACD Bullish Crossover"][i] = f"MACD↑ on {df.index[idx_macd_up].date()}<br>Ret: {ret:.2f}%"

            if "Golden Cross" in selected_bullish:
                idx_gc = last_cross_index(df["50_MA"], df["200_MA"], "up")
                if idx_gc is not None and (not MA_SLOPE_FILTER or (s50 > 0 and s200 >= 0)):
                    ret = compute_return_since(df, idx_gc, "long")
                    heatmap_data["Golden Cross"][i] = ret
                    signals.append((ticker, "bullish", f"✨ Golden Cross — {company} | {df.index[idx_gc].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Golden Cross"][i] = f"50>200 on {df.index[idx_gc].date()}<br>Ret: {ret:.2f}%"

            if "Trend + MACD Bullish" in selected_bullish:
                idx_trend = last_cross_index(df["20_MA"], df["50_MA"], "up")
                idx_macd_up = last_cross_index(df["MACD"], df["MACD_Signal"], "up")
                if idx_trend is not None and idx_macd_up is not None and abs(idx_trend - idx_macd_up) <= COMBO_WINDOW and (not MA_SLOPE_FILTER or (s20 > 0 and s50 >= 0)) and macd_ok:
                    idx = max(idx_trend, idx_macd_up)
                    ret = compute_return_since(df, idx, "long")
                    heatmap_data["Trend + MACD Bullish"][i] = ret
                    signals.append((ticker, "bullish", f"✨ Trend+MACD — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Trend + MACD Bullish"][i] = f"Within {COMBO_WINDOW} bars<br>Ret: {ret:.2f}%"

            if "Ichimoku Bullish" in selected_bullish:
                idx_i_bull = ichimoku_cross_index(df, 'up', ICHIMOKU_REQUIRE_COLOR, ICHIMOKU_REQUIRE_CHIKOU)
                if idx_i_bull is not None:
                    ret = compute_return_since(df, idx_i_bull, "long")
                    heatmap_data["Ichimoku Bullish"][i] = ret
                    signals.append((ticker, "bullish", f"☁️ Ichimoku Breakout — {company} | {df.index[idx_i_bull].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Ichimoku Bullish"][i] = f"Cloud up cross on {df.index[idx_i_bull].date()}<br>Ret: {ret:.2f}%"

            # Bearish (negative)
            if "RSI Overbought (Re-entry)" in selected_bearish:
                idx_rsi_bear = last_threshold_cross_index(df["RSI"], 70.0, 'reenter_below')
                if idx_rsi_bear is not None:
                    ret = compute_return_since(df, idx_rsi_bear, "short")
                    heatmap_data["RSI Overbought (Re-entry)"][i] = -ret
                    signals.append((ticker, "bearish", f"📉 RSI Re-entry <70 — {company} | {df.index[idx_rsi_bear].date()} → {ret:.2f}%"))
                    heatmap_hover_text["RSI Overbought (Re-entry)"][i] = f"RSI↓ <70 on {df.index[idx_rsi_bear].date()}<br>Short Ret: {ret:.2f}%"

            if "MACD Bearish Crossover" in selected_bearish:
                idx_macd_dn = last_cross_index(df["MACD"], df["MACD_Signal"], "down")
                if idx_macd_dn is not None and macd_ok:
                    ret = compute_return_since(df, idx_macd_dn, "short")
                    heatmap_data["MACD Bearish Crossover"][i] = -ret
                    signals.append((ticker, "bearish", f"📉 MACD Cross Down — {company} | {df.index[idx_macd_dn].date()} → {ret:.2f}%"))
                    heatmap_hover_text["MACD Bearish Crossover"][i] = f"MACD↓ on {df.index[idx_macd_dn].date()}<br>Short Ret: {ret:.2f}%"

            if "Death Cross" in selected_bearish:
                idx_dc = last_cross_index(df["50_MA"], df["200_MA"], "down")
                if idx_dc is not None and (not MA_SLOPE_FILTER or (s50 < 0 and s200 <= 0)):
                    ret = compute_return_since(df, idx_dc, "short")
                    heatmap_data["Death Cross"][i] = -ret
                    signals.append((ticker, "bearish", f"💀 Death Cross — {company} | {df.index[idx_dc].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Death Cross"][i] = f"50<200 on {df.index[idx_dc].date()}<br>Short Ret: {ret:.2f}%"

            if "Death Cross + RSI Bearish" in selected_bearish:
                idx_dc = last_cross_index(df["50_MA"], df["200_MA"], "down")
                idx_rsi_bear = last_threshold_cross_index(df["RSI"], 70.0, 'reenter_below')
                if idx_dc is not None and idx_rsi_bear is not None and abs(idx_dc - idx_rsi_bear) <= COMBO_WINDOW and (not MA_SLOPE_FILTER or (s50 < 0 and s200 <= 0)):
                    idx = max(idx_dc, idx_rsi_bear)
                    ret = compute_return_since(df, idx, "short")
                    heatmap_data["Death Cross + RSI Bearish"][i] = -ret
                    signals.append((ticker, "bearish", f"💀 Death+RSI — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Death Cross + RSI Bearish"][i] = f"Within {COMBO_WINDOW} bars<br>Short Ret: {ret:.2f}%"

            if "Ichimoku Bearish" in selected_bearish:
                idx_i_bear = ichimoku_cross_index(df, 'down', ICHIMOKU_REQUIRE_COLOR, ICHIMOKU_REQUIRE_CHIKOU)
                if idx_i_bear is not None:
                    ret = compute_return_since(df, idx_i_bear, "short")
                    heatmap_data["Ichimoku Bearish"][i] = -ret
                    signals.append((ticker, "bearish", f"☁️ Ichimoku Breakdown — {company} | {df.index[idx_i_bear].date()} → {ret:.2f}%"))
                    heatmap_hover_text["Ichimoku Bearish"][i] = f"Cloud down cross on {df.index[idx_i_bear].date()}<br>Short Ret: {ret:.2f}%"
    else:
        st.info("⚠️ Please select at least one strategy from the sidebar to generate signals.")

heatmap_df = pd.DataFrame(heatmap_data, index=TICKERS)
heatmap_hover_df = pd.DataFrame(heatmap_hover_text, index=TICKERS)

with tab1:
    st.markdown("### 📊 Market Overview")
    kpi_ticker = st.selectbox("Select a Ticker to view KPIs", TICKERS, index=0, key="kpi_select")
    kpi_df, _ = fetch_and_process_data(kpi_ticker, "1 year")
    if kpi_df is not None and not kpi_df.empty and len(kpi_df) >= 2:
        latest_price = float(kpi_df['Close'].iloc[-1]); previous_price = float(kpi_df['Close'].iloc[-2])
        current_volume = int(kpi_df['Volume'].iloc[-1])
        low_price = float(kpi_df['Low'].iloc[-1]); high_price = float(kpi_df['High'].iloc[-1])
        avg_volume = float(kpi_df['Volume'].mean())
        change_pct = ((latest_price - previous_price) / previous_price) * 100 if previous_price else 0
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(label=f"Price ({kpi_ticker})", value=f"${latest_price:.2f}", delta=f"{change_pct:.2f}%")
        with col2: st.metric(label="Today's Range", value=f"${low_price:.2f} - ${high_price:.2f}")
        with col3: st.metric(label="Current Volume", value=f"{current_volume:,}")
        with col4: st.metric(label="Average Volume", value=f"{avg_volume:,.0f}")
    else:
        st.warning(f"⚠️ Insufficient data for {kpi_ticker} to calculate KPIs. Please check back later.")

    st.markdown("---")
    st.markdown("### ✅ Current Trade Signals")
    if signals:
        for _, signal_type, msg in signals:
            st.success(msg) if signal_type == 'bullish' else st.error(msg)
    elif selected_strategies:
        st.info("No trade signals at this time for any active strategies.")
    else:
        st.info("Please select strategies from the sidebar to see current trade signals.")

    st.markdown("---")
    st.markdown("### 📊 Strategy Heatmap")
    if not heatmap_df.empty:
        max_abs = np.nanmax(np.abs(heatmap_df.values)) if heatmap_df.size else 1.0
        if not np.isfinite(max_abs) or max_abs == 0: max_abs = 1.0
        max_abs = min(max_abs, float(COLOR_CLAMP))
        colorscale = [[0.0, "rgb(180,0,0)"],[0.5, "rgb(220,220,220)"],[1.0, "rgb(0,180,0)"]]
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=list(heatmap_df.columns),
            y=list(heatmap_df.index),
            colorscale=colorscale,
            zmin=-max_abs, zmax=+max_abs, zmid=0,
            colorbar=dict(title="Return %", ticksuffix="%"),
            text=heatmap_hover_df.values, hoverinfo='text'))
        fig_heatmap.update_layout(title="Active Signals by Strategy and Ticker",
                                  xaxis_title="Strategies", yaxis_title="Tickers",
                                  xaxis=dict(tickangle=45, type='category', constrain='domain'),
                                  yaxis=dict(autorange='reversed', type='category', constrain='domain'),
                                  autosize=True, margin=dict(l=60, r=10, t=60, b=120))
        fig_heatmap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
        fig_heatmap.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

with tab2:
    st.markdown("### 🔎 Detailed Chart Analysis")
    chart_ticker = st.selectbox("Select a Ticker", options=TICKERS, key="chart_ticker")
    if chart_ticker:
        chart_df, error_msg = fetch_and_process_data(chart_ticker, timeframe)
        if chart_df is not None and not chart_df.empty:
            fig_candlestick = go.Figure(data=[
                go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price'),
                go.Scatter(x=chart_df.index, y=chart_df['20_MA'], mode='lines', name='20 MA', line=dict(color='orange', width=2)),
                go.Scatter(x=chart_df.index, y=chart_df['50_MA'], mode='lines', name='50 MA', line=dict(color='blue', width=2)),
                go.Scatter(x=chart_df.index, y=chart_df['200_MA'], mode='lines', name='200 MA', line=dict(color='red', width=2))
            ])
            fig_candlestick.update_layout(title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Price Action (Moving Averages)", xaxis_rangeslider_visible=False, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig_candlestick, use_container_width=True)

            fig_ichimoku = go.Figure(data=[
                go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price'),
                go.Scatter(x=chart_df.index, y=chart_df['tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='red', width=1)),
                go.Scatter(x=chart_df.index, y=chart_df['kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='blue', width=1)),
                go.Scatter(x=chart_df.index, y=chart_df['chikou_span'], mode='lines', name='Chikou Span', line=dict(color='green', width=1)),
                go.Scatter(x=chart_df.index, y=chart_df['senkou_span_a'], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Cloud'),
                go.Scatter(x=chart_df.index, y=chart_df['senkou_span_b'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(255, 0, 0, 0.2)' if chart_df['senkou_span_a'].iloc[-1] > chart_df['senkou_span_b'].iloc[-1] else 'rgba(0, 255, 0, 0.2)', name='Cloud Fill')
            ])
            fig_ichimoku.update_layout(title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Ichimoku Cloud ({timeframe} interval)", xaxis_rangeslider_visible=False, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig_ichimoku, use_container_width=True)
        else:
            st.warning(f"⚠️ No data available for {chart_ticker} at the selected timeframe.")
