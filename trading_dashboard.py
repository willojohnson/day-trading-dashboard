import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh
from datetime import timezone

# =============================================================
# Trading Dashboard – Split Heatmaps + Filters + Leaderboards
# Pre-styling version (reverted), with hover toggle for company names
# + Last Refresh timestamp & data source caption
# + Fresh-trigger borders & Data latency badge
# =============================================================

# --- Tickers to Monitor ---
TICKERS = [
    "NVDA", "AMZN", "META", "TSLA", "SNOW", "AI",
    "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG", "BTSG", "ARDX"
]

# --- Ticker to Company Name Mapping ---
TICKER_NAMES = {
    "NVDA": "NVIDIA Corporation",
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
    "BTSG": "BrightSpring Health Services",
    "ARDX": "Ardelyx Pharmaceuticals",
}

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Strategy Heatmap")

# Keep a persistent “last refresh” text
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = "—"

tab1, tab2 = st.tabs(["Dashboard Overview", "Chart Analysis"])

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=10, max_value=300, value=30, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Interactive Strategy Selectors ---
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

# --- Strategy Filters / Options ---
st.sidebar.markdown("### 🔧 Signal Filters & Options")
MA_SLOPE_FILTER = st.sidebar.checkbox("Require MA slope confirmation (20/50/200)", value=True)
COMBO_WINDOW = st.sidebar.number_input("Combo window (bars)", min_value=0, max_value=10, value=3, step=1)
MACD_ATR_FILTER_ON = st.sidebar.checkbox("Require minimum MACD delta vs Signal (ATR-based)", value=True)
MACD_ATR_MULT = st.sidebar.slider("MACD delta ≥ ATR × multiplier", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
ICHIMOKU_REQUIRE_COLOR = st.sidebar.checkbox("Ichimoku: require bullish/bearish cloud color", value=True)
ICHIMOKU_REQUIRE_CHIKOU = st.sidebar.checkbox("Ichimoku: require Chikou confirmation", value=True)
COLOR_CLAMP = st.sidebar.slider("Heatmap color clamp (abs %)", 10, 100, 50, 5)

# --- View Controls for Split Heatmaps ---
st.sidebar.markdown("### 🧭 Heatmap View Controls")
TRIGGERED_ONLY = st.sidebar.toggle("Show triggered only", value=True)
MIN_RET = st.sidebar.slider("Min Return %", 0.0, 50.0, 3.0, 0.5)
MAX_AGE_DAYS = st.sidebar.slider("Max Signal Age (days)", 1, 180, 30)
ROW_ORDER = st.sidebar.selectbox("Sort rows (tickers) by", ["Best return", "Most recent", "A–Z"], index=0)
COL_ORDER = st.sidebar.selectbox("Sort columns (strategies) by", ["Most triggers", "Best avg return", "Most recent", "A–Z"], index=0)

# --- Hover toggle for company names ---
SHOW_FULLNAME_HOVER = st.sidebar.toggle("Show company names in hover", value=True)

# --- New: Fresh border + Latency controls ---
st.sidebar.markdown("### ✨ Visual Accents")
FRESH_AGE_DAYS = st.sidebar.slider("Fresh border if Age ≤ (days)", 0.5, 7.0, 2.0, 0.5)
LATENCY_ALERT_MIN = st.sidebar.slider("Latency alert ≥ (minutes, intraday only)", 1, 60, 10)

# --- Collapsible Strategy Definitions Section ---
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

# --- Data Fetch & Indicators ---
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
        start_date = end_date - datetime.timedelta(days=5*365)

    if start_date is None:
        return None, "Unsupported timeframe."

    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty or "Close" not in df.columns:
        return None, "No valid data."

    df = df.copy()

    # Indicator Calculations
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ATR (14)
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Ichimoku Cloud
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['chikou_span'] = df['Close'].shift(-26)

    return df, None

# ---------- Helpers (positional indices) ----------

def _nanmask(a: pd.Series, b: pd.Series):
    m = (~a.isna()) & (~b.isna())
    return a[m].values, b[m].values, np.where(m)[0]

def last_cross_index(series_a: pd.Series, series_b: pd.Series, direction: str):
    a, b, posmap = _nanmask(series_a, series_b)
    if len(a) < 2:
        return None
    cross = (a[:-1] <= b[:-1]) & (a[1:] > b[1:]) if direction == 'up' else (a[:-1] >= b[:-1]) & (a[1:] < b[1:])
    idxs = np.where(cross)[0]
    return int(posmap[idxs[-1] + 1]) if idxs.size else None

def last_threshold_cross_index(series: pd.Series, thresh: float, mode: str):
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
    c = close.values[valid_np]; A = a.values[valid_np]; B = b.values[valid_np]
    posmap = np.where(valid_np)[0]

    cloud_top_prev = np.maximum(A[:-1], B[:-1]); cloud_top_now  = np.maximum(A[1:],  B[1:])
    cloud_bot_prev = np.minimum(A[:-1], B[:-1]); cloud_bot_now  = np.minimum(A[1:],  B[1:])

    cross = (c[:-1] <= cloud_top_prev) & (c[1:] > cloud_top_now) if direction == 'up' \
            else (c[:-1] >= cloud_bot_prev) & (c[1:] < cloud_bot_now)

    idxs = np.where(cross)[0]
    if not idxs.size:
        return None

    for k in idxs[::-1]:
        pos = int(posmap[k + 1])
        ok = True
        if require_color:
            if direction == 'up' and not (df['senkou_span_a'].iloc[pos] > df['senkou_span_b'].iloc[pos]): ok = False
            if direction == 'down' and not (df['senkou_span_a'].iloc[pos] < df['senkou_span_b'].iloc[pos]): ok = False
        if ok and require_chikou and pos - 26 >= 0:
            price_back = df['Close'].iloc[pos - 26]; chikou_now = df['chikou_span'].iloc[pos - 26]
            if direction == 'up' and not (chikou_now > price_back): ok = False
            if direction == 'down' and not (chikou_now < price_back): ok = False
        if ok: return pos
    return None

# ---------- Metrics helpers ----------

def compute_return_since(df: pd.DataFrame, trigger_idx: int, side: str) -> float:
    if trigger_idx is None or trigger_idx >= len(df): return 0.0
    entry = float(df['Close'].iloc[trigger_idx]); last = float(df['Close'].iloc[-1])
    if entry <= 0 or last <= 0: return 0.0
    return float(((last/entry - 1.0) if side == 'long' else (entry/last - 1.0)) * 100.0)

def signal_age_days(df: pd.DataFrame, trigger_idx: int) -> float:
    if trigger_idx is None or trigger_idx >= len(df): return np.nan
    end_ts = df.index[-1]; trig_ts = df.index[trigger_idx]
    delta = (end_ts - trig_ts)
    # Support intraday frequency
    return round(delta.total_seconds() / 86400.0, 2)

def slope(series: pd.Series, lookback: int = 5):
    if len(series.dropna()) <= lookback: return 0.0
    return float(series.iloc[-1] - series.iloc[-lookback-1])

def _humanize_secs(s: float) -> str:
    if s is None or not np.isfinite(s): return "—"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m"
    if m: return f"{m}m {sec}s"
    return f"{sec}s"

# ---------- Main logic builds returns + ages + hover + latency ----------

signals = []
heatmap_data = {s: np.zeros(len(TICKERS), dtype=float) for s in all_strategies}
heatmap_age = {s: np.full(len(TICKERS), np.nan) for s in all_strategies}
heatmap_hover = {s: [""] * len(TICKERS) for s in all_strategies}

# Track worst (maximum) latency across all fetched tickers
worst_latency_sec = None

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

            # --- latency measurement (most recent bar recency) ---
            try:
                now = datetime.datetime.now(timezone.utc).astimezone()
                # yfinance index may be tz-aware or naive; coerce to local tz
                last_ts = df.index[-1]
                if last_ts.tzinfo is None:
                    # assume localize to UTC then convert to local (best-effort)
                    last_ts = last_ts.replace(tzinfo=timezone.utc).astimezone()
                latency = (now - last_ts).total_seconds()
                if (worst_latency_sec is None) or (latency > worst_latency_sec):
                    worst_latency_sec = latency
            except Exception:
                pass

            s20 = slope(df['20_MA']); s50 = slope(df['50_MA']); s200 = slope(df['200_MA'])
            macd_ok = True
            if MACD_ATR_FILTER_ON:
                macd_ok = df['ATR'].iloc[-1] <= 0 or abs(df['MACD_Hist'].iloc[-1]) >= (MACD_ATR_MULT * df['ATR'].iloc[-1])

            # Bullish
            if "Trend Trading" in selected_bullish:
                idx = last_cross_index(df['20_MA'], df['50_MA'], 'up')
                if idx is not None and (not MA_SLOPE_FILTER or (s20 > 0 and s50 >= 0)):
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['Trend Trading'][i] = ret; heatmap_age['Trend Trading'][i] = age
                    signals.append((ticker, 'bullish', f"📈 Trend Up — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Trend Trading'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\n20>50 on {df.index[idx].date()}\nRet: {ret:.2f}%  |  Age: {age}d"

            if "RSI Oversold (Re-entry)" in selected_bullish:
                idx = last_threshold_cross_index(df['RSI'], 30.0, 'reenter_above')
                if idx is not None:
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['RSI Oversold (Re-entry)'][i] = ret; heatmap_age['RSI Oversold (Re-entry)'][i] = age
                    signals.append((ticker, 'bullish', f"📈 RSI Re-entry >30 — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['RSI Oversold (Re-entry)'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nRSI↑ >30 on {df.index[idx].date()}\nRet: {ret:.2f}%  |  Age: {age}d"

            if "MACD Bullish Crossover" in selected_bullish:
                idx = last_cross_index(df['MACD'], df['MACD_Signal'], 'up')
                if idx is not None and macd_ok:
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['MACD Bullish Crossover'][i] = ret; heatmap_age['MACD Bullish Crossover'][i] = age
                    signals.append((ticker, 'bullish', f"📈 MACD Cross Up — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['MACD Bullish Crossover'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nMACD↑ on {df.index[idx].date()}\nRet: {ret:.2f}%  |  Age: {age}d"

            if "Golden Cross" in selected_bullish:
                idx = last_cross_index(df['50_MA'], df['200_MA'], 'up')
                if idx is not None and (not MA_SLOPE_FILTER or (s50 > 0 and s200 >= 0)):
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['Golden Cross'][i] = ret; heatmap_age['Golden Cross'][i] = age
                    signals.append((ticker, 'bullish', f"✨ Golden Cross — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Golden Cross'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\n50>200 on {df.index[idx].date()}\nRet: {ret:.2f}%  |  Age: {age}d"

            if "Trend + MACD Bullish" in selected_bullish:
                idx1 = last_cross_index(df['20_MA'], df['50_MA'], 'up'); idx2 = last_cross_index(df['MACD'], df['MACD_Signal'], 'up')
                if idx1 is not None and idx2 is not None and abs(idx1 - idx2) <= COMBO_WINDOW and (not MA_SLOPE_FILTER or (s20 > 0 and s50 >= 0)) and macd_ok:
                    idx = max(idx1, idx2)
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['Trend + MACD Bullish'][i] = ret; heatmap_age['Trend + MACD Bullish'][i] = age
                    signals.append((ticker, 'bullish', f"✨ Trend+MACD — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Trend + MACD Bullish'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nWithin {COMBO_WINDOW} bars\nRet: {ret:.2f}%  |  Age: {age}d"

            if "Ichimoku Bullish" in selected_bullish:
                idx = ichimoku_cross_index(df, 'up', ICHIMOKU_REQUIRE_COLOR, ICHIMOKU_REQUIRE_CHIKOU)
                if idx is not None:
                    ret = compute_return_since(df, idx, 'long'); age = signal_age_days(df, idx)
                    heatmap_data['Ichimoku Bullish'][i] = ret; heatmap_age['Ichimoku Bullish'][i] = age
                    signals.append((ticker, 'bullish', f"☁️ Ichimoku Breakout — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Ichimoku Bullish'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nCloud up cross on {df.index[idx].date()}\nRet: {ret:.2f}%  |  Age: {age}d"

            # Bearish (negative)
            if "RSI Overbought (Re-entry)" in selected_bearish:
                idx = last_threshold_cross_index(df['RSI'], 70.0, 'reenter_below')
                if idx is not None:
                    ret = compute_return_since(df, idx, 'short'); age = signal_age_days(df, idx)
                    heatmap_data['RSI Overbought (Re-entry)'][i] = -ret; heatmap_age['RSI Overbought (Re-entry)'][i] = age
                    signals.append((ticker, 'bearish', f"📉 RSI Re-entry <70 — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['RSI Overbought (Re-entry)'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nRSI↓ <70 on {df.index[idx].date()}\nShort Ret: {ret:.2f}%  |  Age: {age}d"

            if "MACD Bearish Crossover" in selected_bearish:
                idx = last_cross_index(df['MACD'], df['MACD_Signal'], 'down')
                if idx is not None and macd_ok:
                    ret = compute_return_since(df, idx, 'short'); age = signal_age_days(df, idx)
                    heatmap_data['MACD Bearish Crossover'][i] = -ret; heatmap_age['MACD Bearish Crossover'][i] = age
                    signals.append((ticker, 'bearish', f"📉 MACD Cross Down — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['MACD Bearish Crossover'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nMACD↓ on {df.index[idx].date()}\nShort Ret: {ret:.2f}%  |  Age: {age}d"

            if "Death Cross" in selected_bearish:
                idx = last_cross_index(df['50_MA'], df['200_MA'], 'down')
                if idx is not None and (not MA_SLOPE_FILTER or (s50 < 0 and s200 <= 0)):
                    ret = compute_return_since(df, idx, 'short'); age = signal_age_days(df, idx)
                    heatmap_data['Death Cross'][i] = -ret; heatmap_age['Death Cross'][i] = age
                    signals.append((ticker, 'bearish', f"💀 Death Cross — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Death Cross'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\n50<200 on {df.index[idx].date()}\nShort Ret: {ret:.2f}%  |  Age: {age}d"

            if "Death Cross + RSI Bearish" in selected_bearish:
                idx1 = last_cross_index(df['50_MA'], df['200_MA'], 'down'); idx2 = last_threshold_cross_index(df['RSI'], 70.0, 'reenter_below')
                if idx1 is not None and idx2 is not None and abs(idx1 - idx2) <= COMBO_WINDOW and (not MA_SLOPE_FILTER or (s50 < 0 and s200 <= 0)):
                    idx = max(idx1, idx2)
                    ret = compute_return_since(df, idx, 'short'); age = signal_age_days(df, idx)
                    heatmap_data['Death Cross + RSI Bearish'][i] = -ret; heatmap_age['Death Cross + RSI Bearish'][i] = age
                    signals.append((ticker, 'bearish', f"💀 Death+RSI — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Death Cross + RSI Bearish'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nWithin {COMBO_WINDOW} bars\nShort Ret: {ret:.2f}%  |  Age: {age}d"

            if "Ichimoku Bearish" in selected_bearish:
                idx = ichimoku_cross_index(df, 'down', ICHIMOKU_REQUIRE_COLOR, ICHIMOKU_REQUIRE_CHIKOU)
                if idx is not None:
                    ret = compute_return_since(df, idx, 'short'); age = signal_age_days(df, idx)
                    heatmap_data['Ichimoku Bearish'][i] = -ret; heatmap_age['Ichimoku Bearish'][i] = age
                    signals.append((ticker, 'bearish', f"☁️ Ichimoku Breakdown — {company} | {df.index[idx].date()} → {ret:.2f}%"))
                    heatmap_hover['Ichimoku Bearish'][i] = f"{ticker} — {company if SHOW_FULLNAME_HOVER else ''}\nCloud down cross on {df.index[idx].date()}\nShort Ret: {ret:.2f}%  |  Age: {age}d"
    else:
        st.info("⚠️ Please select at least one strategy from the sidebar to generate signals.")

# --- Build DataFrames ---
heatmap_df = pd.DataFrame(heatmap_data, index=TICKERS)
ages_df = pd.DataFrame(heatmap_age, index=TICKERS)
hover_df = pd.DataFrame(heatmap_hover, index=TICKERS)

# Convenience matrices for split views
bull_df = heatmap_df.clip(lower=0)
bear_df = (-heatmap_df.clip(upper=0)).abs()  # convert to positive intensity for bear view

# --- Header meta (last refresh + source + latency) ---
local_ts = datetime.datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
st.session_state.last_refresh = local_ts

latency_badge = ""
if timeframe in ["5m", "15m", "30m", "1h"]:
    # show worst-case latency across tickers
    h = _humanize_secs(worst_latency_sec) if worst_latency_sec is not None else "—"
    if worst_latency_sec is not None and worst_latency_sec >= LATENCY_ALERT_MIN * 60:
        latency_badge = f" • ⚠️ Data latency: {h}"
    else:
        latency_badge = f" • ⏱️ Data latency: {h}"

st.caption(
    f"**Last refresh:** {st.session_state.last_refresh}  •  "
    f"**Source:** Yahoo Finance via yfinance (interval driven by timeframe)  •  "
    f"**Auto-refresh:** every {refresh_rate}s{latency_badge}"
)

# --- DASHBOARD OVERVIEW TAB ---
with tab1:
    # Summary strip
    total_bull = int((bull_df > 0).sum().sum()); total_bear = int((bear_df > 0).sum().sum())
    c1, c2, c3 = st.columns([1,1,2])
    c1.success(f"Bullish: {total_bull} active cells")
    c2.error(f"Bearish: {total_bear} active cells")

    # Controls repeated here (nice UX):
    st.markdown("### 📊 Split Strategy Heatmaps")
    cmin, cage, ctrigger = st.columns([1,1,1])
    with cmin: min_ret_local = st.slider("Min return %", 0.0, 50.0, MIN_RET, 0.5, key="min_ret_local")
    with cage: max_age_local = st.slider("Max age (days)", 1, 180, MAX_AGE_DAYS, key="max_age_local")
    with ctrigger: triggered_only_local = st.toggle("Triggered only", TRIGGERED_ONLY, key="triggered_only_local")

    # Filtering masks
    def filter_matrix(values: pd.DataFrame, ages: pd.DataFrame, min_ret: float, max_age: int, triggered_only: bool):
        Z = values.copy()
        A = ages.reindex_like(values)
        # Drop by min return
        Z = Z.where(Z >= min_ret, np.nan)
        # Drop by max age
        Z = Z.where((A <= max_age) | (~np.isfinite(A)), np.nan)
        if triggered_only:
            Z = Z.where(np.isfinite(Z))  # keep NaNs for non-triggered
        return Z

    bull_plot = filter_matrix(bull_df, ages_df, min_ret_local, max_age_local, triggered_only_local)
    bear_plot = filter_matrix(bear_df, ages_df, min_ret_local, max_age_local, triggered_only_local)

    # Ordering helpers
    def order_rows(df_plot: pd.DataFrame, ages: pd.DataFrame, how: str):
        if how == "Best return":
            score = df_plot.max(axis=1).fillna(0)
            return df_plot.loc[score.sort_values(ascending=False).index], ages.loc[score.sort_values(ascending=False).index]
        if how == "Most recent":
            age_min = ages.where(np.isfinite(df_plot)).min(axis=1).fillna(1e9)
            return df_plot.loc[age_min.sort_values(ascending=True).index], ages.loc[age_min.sort_values(ascending=True).index]
        return df_plot.sort_index(), ages.sort_index()

    def order_cols(df_plot: pd.DataFrame, ages: pd.DataFrame, how: str):
        if how == "Most triggers":
            score = (np.isfinite(df_plot)).sum(axis=0)
            return df_plot.loc[:, score.sort_values(ascending=False).index], ages.loc[:, score.sort_values(ascending=False).index]
        if how == "Best avg return":
            score = df_plot.mean(axis=0, skipna=True).fillna(0)
            return df_plot.loc[:, score.sort_values(ascending=False).index], ages.loc[:, score.sort_values(ascending=False).index]
        if how == "Most recent":
            age_min = ages.where(np.isfinite(df_plot)).min(axis=0).fillna(1e9)
            return df_plot.loc[:, age_min.sort_values(ascending=True).index], ages.loc[:, age_min.sort_values(ascending=True).index]
        return df_plot, ages

    # Apply ordering
    bull_plot, ages_bull = order_rows(bull_plot, ages_df, ROW_ORDER)
    bull_plot, ages_bull = order_cols(bull_plot, ages_bull, COL_ORDER)
    bear_plot, ages_bear = order_rows(bear_plot, ages_df, ROW_ORDER)
    bear_plot, ages_bear = order_cols(bear_plot, ages_bear, COL_ORDER)

    col_bull, col_bear = st.columns(2)

    def render_heatmap(df_plot: pd.DataFrame, ages_plot: pd.DataFrame, title: str, palette: str, side: str):
        if df_plot.empty:
            st.info("No data to display.")
            return

        # Build per-cell hovertext
        hover = []
        fresh_x = []  # column names for fresh cells
        fresh_y = []  # row names (tickers) for fresh cells

        for r in df_plot.index:
            row = []
            for c in df_plot.columns:
                val = df_plot.loc[r, c]
                age = ages_plot.loc[r, c]
                if np.isfinite(val):
                    comp = f"{TICKER_NAMES.get(r, r)}<br>" if SHOW_FULLNAME_HOVER else ""
                    row.append(f"{r} — {c}<br>{comp}Return: {val:.2f}%<br>Age: {age if np.isfinite(age) else '—'}d")
                    # mark fresh cells (bold border overlay)
                    if np.isfinite(age) and age <= FRESH_AGE_DAYS:
                        fresh_x.append(c)
                        fresh_y.append(r)
                else:
                    row.append("")
            hover.append(row)

        fig = go.Figure()

        # Base heatmap
        fig.add_trace(go.Heatmap(
            z=df_plot.values,
            x=list(df_plot.columns),
            y=list(df_plot.index),
            colorscale=palette,
            zmin=0, zmax=float(COLOR_CLAMP),
            hoverinfo='text', text=hover,
            showscale=True,
            colorbar=dict(title=("Return %" if side=='bull' else "Short Ret %"), ticksuffix="%")
        ))

        # Overlay: square markers with transparent fill + visible border to highlight fresh signals
        if len(fresh_x):
            fig.add_trace(go.Scatter(
                x=fresh_x,
                y=fresh_y,
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=28,                 # size in px; looks good for typical layout
                    color="rgba(0,0,0,0)",   # transparent fill
                    line=dict(color="black", width=2)
                ),
                hoverinfo="skip",
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Strategies", yaxis_title="Tickers",
            xaxis=dict(tickangle=45, type='category', constrain='domain'),
            yaxis=dict(autorange='reversed', type='category', constrain='domain'),
            margin=dict(l=60, r=10, t=60, b=120),
            height=520
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.15)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.15)')
        st.plotly_chart(fig, use_container_width=True)

    with col_bull:
        render_heatmap(bull_plot, ages_bull, "Bullish Signals", 'Greens', 'bull')
    with col_bear:
        render_heatmap(bear_plot, ages_bear, "Bearish Signals", 'Reds', 'bear')

    st.markdown("---")

    # Leaderboards
    def melt_active(df_values: pd.DataFrame, df_ages: pd.DataFrame, side: str):
        out = []
        for t in df_values.index:
            for s in df_values.columns:
                v = df_values.loc[t, s]
                if np.isfinite(v) and v > 0:
                    out.append({
                        'Ticker': t,
                        'Strategy': s,
                        'Return %': v if side=='bull' else -v,
                        'Age (d)': df_ages.loc[t, s]
                    })
        return pd.DataFrame(out)

    bull_long = melt_active(bull_df, ages_df, 'bull')
    bear_long = melt_active(bear_df, ages_df, 'bear')

    cL, cR = st.columns(2)
    if not bull_long.empty:
        bull_top = bull_long.sort_values(['Return %', 'Age (d)'], ascending=[False, True]).head(12)
        cL.subheader("Top Bullish")
        cL.dataframe(bull_top, use_container_width=True, hide_index=True)
    else:
        cL.info("No bullish signals to rank.")

    if not bear_long.empty:
        bear_top = bear_long.sort_values(['Return %', 'Age (d)'], ascending=[True, True]).head(12)
        cR.subheader("Top Bearish")
        cR.dataframe(bear_top, use_container_width=True, hide_index=True)
    else:
        cR.info("No bearish signals to rank.")

# --- CHART ANALYSIS TAB ---
with tab2:
    st.markdown("### 🔎 Detailed Chart Analysis")
    chart_ticker = st.selectbox("Select a Ticker", options=TICKERS, key="chart_ticker")

    def fetch_wrap(ticker):
        return fetch_and_process_data(ticker, timeframe)

    if chart_ticker:
        chart_df, error_msg = fetch_wrap(chart_ticker)
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

            fig_ichimoku.update_layout(title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Ichimoku Cloud ({timeframe} interval)", xaxis_rangeslider_visible=False, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

            st.plotly_chart(fig_ichimoku, use_container_width=True)
        else:
            st.warning(f"⚠️ No data available for {chart_ticker} at the selected timeframe.")
