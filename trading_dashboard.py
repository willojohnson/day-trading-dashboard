import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# --- Tickers to Monitor ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"]

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
    "DDOG": "Datadog Inc."
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
bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought"]

selected_bullish = st.sidebar.multiselect("📈 Bullish Strategies", bullish_strategies)
selected_bearish = st.sidebar.multiselect("📉 Bearish Strategies", bearish_strategies)

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown(""")
**Trend Trading**: 20MA > 50MA  
**RSI Overbought**: RSI > 70  
**RSI Oversold**: RSI < 30  
**MACD Bullish Crossover**: MACD crosses above Signal  
**MACD Bearish Crossover**: MACD crosses below Signal  

# --- Signal Detection ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=5)
end = now

signals = []
heatmap_data = []

for ticker in TICKERS:
    company = TICKER_NAMES.get(ticker, ticker)
    st.subheader(f"📈 {ticker} — {company}")
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")
        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data for {ticker}.")
            continue

        # Indicators
        df['20_MA'] = df['Close'].rolling(20).mean()
        df['50_MA'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['Close'].ewm(span=3, adjust=False).mean()
        exp2 = df['Close'].ewm(span=10, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=16, adjust=False).mean()

        # Signal Matrix Row
        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = 0

        # Bullish Strategies
        if "Trend Trading" in selected_bullish and df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
            signals.append((ticker, f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = 1

        if "RSI Oversold" in selected_bullish and df['RSI'].iloc[-1] < 30:
            signals.append((ticker, f"📈 Bullish - RSI Oversold — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Oversold"] = 1

        if "MACD Bullish Crossover" in selected_bullish:
            if df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                signals.append((ticker, f"📈 Bullish - MACD Bullish Crossover — {company}"))
                heatmap_row["MACD Bullish Crossover"] = 1

        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and df['RSI'].iloc[-1] > 70:
            signals.append((ticker, f"📉 Bearish - RSI Overbought — {company} (RSI={df['RSI'].iloc[-1]:.1f})"))
            heatmap_row["RSI Overbought"] = 1

        if "MACD Bearish Crossover" in selected_bearish:
            if df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
                signals.append((ticker, f"📉 Bearish - MACD Bearish Crossover — {company}"))
                heatmap_row["MACD Bearish Crossover"] = 1

        heatmap_data.append(heatmap_row)

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")

# --- Signal Display ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, msg in signals:
        st.success(msg)
else:
    st.info("No trade signals at this time.")

# --- Heatmap Matrix + Visual ---
if heatmap_data:
    st.markdown("### 🧭 Strategy Signal Matrix")

    heatmap_df = pd.DataFrame(heatmap_data)
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
