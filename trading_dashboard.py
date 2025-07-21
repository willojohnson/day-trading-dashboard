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
bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold"]
# Added "Death Cross" to bearish strategies
bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Death Cross"]

# Removed st.multiselect and set selected_bullish/bearish to include all strategies
selected_bullish = bullish_strategies
selected_bearish = bearish_strategies

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("""
**Trend Trading**: 20MA > 50MA
**RSI Overbought**: RSI > 70
**RSI Oversold**: RSI < 30
**MACD Bullish Crossover**: MACD crosses above Signal
**MACD Bearish Crossover**: MACD crosses below Signal
**Death Cross**: 50MA crosses below 200MA
""")

# --- Signal Detection ---
now = datetime.datetime.now()
# Increased historical data fetch for 200MA. Max for 5m interval is typically 60 days.
start = now - datetime.timedelta(days=60)
end = now

signals = []
heatmap_data = []

st.subheader("⚙️ Processing Data and Generating Signals...") # Added a general processing message

for ticker in TICKERS:
    company = TICKER_NAMES.get(ticker, ticker)
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")
        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data for {ticker} ({company}). Skipping...")
            continue

        # Ensure enough data for calculations (especially for 200-period MA)
        if len(df) < 200: # Need at least 200 periods for 200MA
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators (requires 200 bars). Skipping...")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = 0
            heatmap_data.append(heatmap_row)
            continue

        # Indicators
        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['200_MA'] = df['Close'].rolling(window=200).mean() # New: 200 MA for Death Cross

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean() # Using EWM for RSI
        avg_loss = loss.ewm(com=13, adjust=False).mean() # Using EWM for RSI
        
        # Handle division by zero for rs in RSI calculation
        rs = avg_gain / avg_loss.replace(0, 1e-9) 
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean() # Standard MACD fast period
        exp2 = df['Close'].ewm(span=26, adjust=False).mean() # Standard MACD slow period
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean() # Standard MACD signal period

        # Signal Matrix Row Initialization
        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = 0 # Initialize all strategy columns to 0 (no signal)

        # Check for NaN values at the end of the dataframe for indicators
        if pd.isna(df['20_MA'].iloc[-1]) or pd.isna(df['50_MA'].iloc[-1]) or \
           pd.isna(df['200_MA'].iloc[-1]) or pd.isna(df['RSI'].iloc[-1]) or \
           pd.isna(df['MACD'].iloc[-1]) or pd.isna(df['MACD_Signal'].iloc[-1]):
            st.info(f"ℹ️ Not enough complete indicator data for {ticker} ({company}). Skipping strategy checks for this ticker.")
            heatmap_data.append(heatmap_row) # Still add to heatmap data even if no signals
            continue

        # Bullish Strategies
        if "Trend Trading" in selected_bullish and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = 1

        if "RSI Oversold" in selected_bullish and df['RSI'].iloc[-1] < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Oversold"] = 1

        if "MACD Bullish Crossover" in selected_bullish:
            if len(df) >= 2 and \
               df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and \
               df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
                heatmap_row["MACD Bullish Crossover"] = 1

        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and df['RSI'].iloc[-1] > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Overbought"] = 1

        if "MACD Bearish Crossover" in selected_bearish:
            if len(df) >= 2 and \
               df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and \
               df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
                signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
                heatmap_row["MACD Bearish Crossover"] = 1
        
        # New: Death Cross Strategy
        if "Death Cross" in selected_bearish:
            # Ensure enough data points for crossover check (at least two bars) and MAs are not NaN
            if len(df) >= 2 and \
               pd.notna(df['50_MA'].iloc[-2]) and pd.notna(df['200_MA'].iloc[-2]) and \
               pd.notna(df['50_MA'].iloc[-1]) and pd.notna(df['200_MA'].iloc[-1]):
                
                # Check for crossover: 50MA was above 200MA, and now 50MA is below 200MA
                if (df['50_MA'].iloc[-2] > df['200_MA'].iloc[-2]) and \
                   (df['50_MA'].iloc[-1] < df['200_MA'].iloc[-1]):
                    signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
                    heatmap_row["Death Cross"] = 1

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
    # Ensure all strategies are columns even if no signals were generated for them
    for strat in bullish_strategies + bearish_strategies:
        if strat not in heatmap_df.columns:
            heatmap_df[strat] = 0

    heatmap_df["Bullish Total"] = heatmap_df[bullish_strategies].sum(axis=1)
    heatmap_df["Bearish Total"] = heatmap_df[bearish_strategies].sum(axis=1)

    # Updated ordered_cols to include "Death Cross"
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
            return 0.0  # neutral gray
        elif strat in bullish_strategies:
            return 1.0  # green
        elif strat in bearish_strategies:
            return -1.0  # red

    matrix_scaled = matrix.copy()
    for col in matrix.columns:
        matrix_scaled[col] = matrix[col].apply(lambda v: custom_color(v, col))

    # Define diverging colors: red → gray → green
    custom_colorscale = [
        [0.0, "lightcoral"], # Corresponds to -1.0 (Bearish)
        [0.5, "#eeeeee"],   # Corresponds to 0.0 (Neutral)
        [1.0, "lightgreen"] # Corresponds to 1.0 (Bullish)
    ]

    fig = px.imshow(
        matrix_scaled,
        color_continuous_scale=custom_colorscale,
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)
