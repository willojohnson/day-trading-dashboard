import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# --- Tickers to Monitor ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG", "BTSG"]

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

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=30, max_value=300, value=60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Strategy Selectors ---
bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Golden Cross", "Trend + MACD Bullish", "Golden Cross + Volume"]
bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Death Cross", "Death Cross + RSI Bearish", "Death Cross + Volume"]

selected_bullish = bullish_strategies
selected_bearish = bearish_strategies

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("**Trend Trading**: 20MA > 50MA")
st.sidebar.markdown("**RSI Overbought**: RSI > 70")
st.sidebar.markdown("**RSI Oversold**: RSI < 30")
st.sidebar.markdown("**MACD Bullish Crossover**: MACD crosses above Signal")
st.sidebar.markdown("**MACD Bearish Crossover**: MACD crosses below Signal")
st.sidebar.markdown("**Death Cross**: 50MA crosses below 200MA")
st.sidebar.markdown("**Golden Cross**: 50MA crosses above 200MA")
st.sidebar.markdown("---")
st.sidebar.markdown("**Trend + MACD Bullish**: 20MA > 50MA AND MACD Bullish Crossover")
st.sidebar.markdown("**Death Cross + RSI Bearish**: 50MA < 200MA AND RSI > 70")
st.sidebar.markdown("**Golden Cross + Volume**: Golden Cross AND Above-Avg Volume")
st.sidebar.markdown("**Death Cross + Volume**: Death Cross AND Above-Avg Volume")


# --- Signal Detection ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=60)
end = now

signals = []
heatmap_data = []

st.subheader("⚙️ Processing Data and Generating Signals...")

for ticker in TICKERS:
    company = TICKER_NAMES.get(ticker, ticker)
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")
        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data for {ticker} ({company}). Skipping...")
            continue

        if len(df) < 200:
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators (requires 200 bars). Skipping...")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = 0
            heatmap_data.append(heatmap_row)
            continue

        # Indicators
        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['200_MA'] = df['Close'].rolling(window=200).mean()
        df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Signal Matrix Row Initialization
        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = 0

        # Check for NaN values at the end of the dataframe for indicators
        if len(df) < 2 or \
           pd.isna(df['20_MA'].iloc[-1]) or pd.isna(df['50_MA'].iloc[-1]) or \
           pd.isna(df['200_MA'].iloc[-1]) or pd.isna(df['RSI'].iloc[-1]) or \
           pd.isna(df['MACD'].iloc[-1]) or pd.isna(df['MACD_Signal'].iloc[-1]) or \
           pd.isna(df['MACD'].iloc[-2]) or pd.isna(df['MACD_Signal'].iloc[-2]) or \
           pd.isna(df['50_MA'].iloc[-2]) or pd.isna(df['200_MA'].iloc[-2]) or \
           pd.isna(df['Avg_Volume'].iloc[-1]) or pd.isna(df['Volume'].iloc[-1]):
            st.info(f"ℹ️ Not enough complete indicator data for {ticker} ({company}). Skipping strategy checks for this ticker.")
            heatmap_data.append(heatmap_row)
            continue

        # Reusable conditions for clarity in confirmation strategies
        # --- FIX: Use .item() to ensure scalar values ---
        current_volume = df['Volume'].iloc[-1].item()
        avg_volume = df['Avg_Volume'].iloc[-1].item()
        is_above_avg_volume = current_volume > 1.2 * avg_volume # This will now be a pure boolean

        # MACD Bullish Crossover Condition
        macd_bullish_crossover = (df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and \
                                  df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1])

        # Golden Cross Condition
        golden_cross = (df['50_MA'].iloc[-2] < df['200_MA'].iloc[-2]) and \
                       (df['50_MA'].iloc[-1] > df['200_MA'].iloc[-1])

        # MACD Bearish Crossover Condition
        macd_bearish_crossover = (df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and \
                                  df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1])

        # Death Cross Condition
        death_cross = (df['50_MA'].iloc[-2] > df['200_MA'].iloc[-2]) and \
                      (df['50_MA'].iloc[-1] < df['200_MA'].iloc[-1])


        # Bullish Strategies
        if "Trend Trading" in selected_bullish and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = 1

        if "RSI Oversold" in selected_bullish and df['RSI'].iloc[-1] < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Oversold"] = 1

        if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
            signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
            heatmap_row["MACD Bullish Crossover"] = 1
        
        if "Golden Cross" in selected_bullish and golden_cross:
            signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
            heatmap_row["Golden Cross"] = 1

        # --- Bullish Confirmation Strategies ---
        if "Trend + MACD Bullish" in selected_bullish:
            if (df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_row["Trend + MACD Bullish"] = 1

        if "Golden Cross + Volume" in selected_bullish:
            if golden_cross and is_above_avg_volume:
                signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross + Volume Confirmed — {company}"))
                heatmap_row["Golden Cross + Volume"] = 1


        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and df['RSI'].iloc[-1] > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Overbought"] = 1

        if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
            signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
            heatmap_row["MACD Bearish Crossover"] = 1
        
        if "Death Cross" in selected_bearish and death_cross:
            signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
            heatmap_row["Death Cross"] = 1
        
        # --- Bearish Confirmation Strategies ---
        if "Death Cross + RSI Bearish" in selected_bearish:
            if death_cross and (df['RSI'].iloc[-1] > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_row["Death Cross + RSI Bearish"] = 1
        
        if "Death Cross + Volume" in selected_bearish:
            if death_cross and is_above_avg_volume:
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + Volume Confirmed — {company}"))
                heatmap_row["Death Cross + Volume"] = 1


        heatmap_data.append(heatmap_row)

    except Exception as e:
        st.error(f"❌ Error processing {ticker} ({company}): {e}")

# --- Signal Display ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, signal_type, msg in signals:
        if signal_type == "bullish":
            st.success(msg)
        elif signal_type == "bearish":
            st.error(msg)
else:
    st.info("No trade signals at this time for any active strategies.")

# --- Heatmap Matrix + Visual ---
if heatmap_data:
    st.markdown("### 🧭 Strategy Signal Matrix")

    heatmap_df = pd.DataFrame(heatmap_data)
    for strat in bullish_strategies + bearish_strategies:
        if strat not in heatmap_df.columns:
            heatmap_df[strat] = 0

    heatmap_df["Bullish Total"] = heatmap_df[bullish_strategies].sum(axis=1)
    heatmap_df["Bearish Total"] = heatmap_df[bearish_strategies].sum(axis=1)

    ordered_cols = ["Label"] + bullish_strategies + ["Bullish Total"] + bearish_strategies + ["Bearish Total"]
    heatmap_df = heatmap_df[ordered_cols]

    st.dataframe(
        heatmap_df.style
        .highlight_max(axis=0, subset=["Bullish Total"], color="lightgreen")
        .highlight_max(axis=0, subset=["Bearish Total"], color="salmon")
    )

    # --- Combined Heatmap Visualization ---
    st.markdown("### 🔥 Strategy Activation Heatmap")

    matrix = heatmap_df.set_index("Label")[bullish_strategies + bearish_strategies]

    def custom_color(val, strat):
        if val == 0:
            return 0.0
        elif strat in bullish_strategies:
            return 1.0
        elif strat in bearish_strategies:
            return -1.0

    matrix_scaled = matrix.copy()
    for col in matrix.columns:
        matrix_scaled[col] = matrix[col].apply(lambda v: custom_color(v, col))

    custom_colorscale = [
        [0.0, "lightcoral"],
        [0.5, "#eeeeee"],
        [1.0, "lightgreen"]
    ]

    fig = px.imshow(
        matrix_scaled,
        color_continuous_scale=custom_colorscale,
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)
