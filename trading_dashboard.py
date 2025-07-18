import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# --- Tickers & Company Mapping ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"]
TICKER_NAMES = {
    "NVDA": "NVIDIA Corporation", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "META": "Meta Platforms Inc.", "TSLA": "Tesla Inc.",
    "SNOW": "Snowflake Inc.", "AI": "C3.ai Inc.", "AMD": "Advanced Micro Devices Inc.",
    "BBAI": "BigBear.ai Holdings Inc.", "SOUN": "SoundHound AI Inc.", "CRSP": "CRISPR Therapeutics AG",
    "TSM": "Taiwan Semiconductor", "DDOG": "Datadog Inc."
}

# --- Strategies ---
bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Bollinger Breakout"]
bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Bollinger Rejection"]

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard (All Strategies Active)")
refresh_rate = st.sidebar.slider("Refresh every N seconds", 30, 300, 60, 10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Time Range ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=5)
end = now

# --- Processing Loop ---
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

        # --- Indicators ---
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
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

        # --- Signal Matrix Row ---
        row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            row[strat] = 0

        # --- Bullish Checks ---
        if df['20_MA'].iloc[-1] > df['50_MA'].iloc[-1]:
            signals.append(f"📈 Trend Trading — {company}")
            row["Trend Trading"] = 1

        if df['RSI'].iloc[-1] < 30:
            signals.append(f"📈 RSI Oversold — {company} (RSI={df['RSI'].iloc[-1]:.1f})")
            row["RSI Oversold"] = 1

        if df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            signals.append(f"📈 MACD Bullish Crossover — {company}")
            row["MACD Bullish Crossover"] = 1

        if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
            signals.append(f"📈 Bollinger Breakout — {company}")
            row["Bollinger Breakout"] = 1

        # --- Bearish Checks ---
        if df['RSI'].iloc[-1] > 70:
            signals.append(f"📉 RSI Overbought — {company} (RSI={df['RSI'].iloc[-1]:.1f})")
            row["RSI Overbought"] = 1

        if df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
            signals.append(f"📉 MACD Bearish Crossover — {company}")
            row["MACD Bearish Crossover"] = 1

        if df['High'].iloc[-1] >= df['BB_Upper'].iloc[-1] and df['Close'].iloc[-1] < df['BB_Upper'].iloc[-1]:
            signals.append(f"📉 Bollinger Rejection — {company}")
            row["Bollinger Rejection"] = 1

        heatmap_data.append(row)

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")
# --- Signal Display ---
st.markdown("### ✅ Strategy Matches Across All Tickers")
if signals:
    for msg in signals:
        st.success(msg)
else:
    st.info("No trade signals triggered.")

# --- Strategy Signal Matrix & Heatmap ---
if heatmap_data:
    st.markdown("### 🧭 Strategy Signal Matrix")

    df_hm = pd.DataFrame(heatmap_data)
    df_hm["Bullish Total"] = df_hm[bullish_strategies].sum(axis=1)
    df_hm["Bearish Total"] = df_hm[bearish_strategies].sum(axis=1)

    ordered = ["Label"] + bullish_strategies + ["Bullish Total"] + bearish_strategies + ["Bearish Total"]
    df_hm = df_hm[ordered]

    st.dataframe(
        df_hm.style
        .highlight_max(axis=0, subset=["Bullish Total"], color="lightgreen")
        .highlight_max(axis=0, subset=["Bearish Total"], color="salmon")
    )

    # --- Combined Heatmap Visualization ---
    st.markdown("### 🔥 Strategy Activation Heatmap")

    matrix = df_hm.set_index("Label")[bullish_strategies + bearish_strategies]

    def custom_color(val, strat):
        if val == 0:
            return 0.0  # neutral gray
        elif strat in bullish_strategies:
            return 1.0  # green
        elif strat in bearish_strategies:
            return -1.0  # red

    scaled = matrix.copy()
    for col in scaled.columns:
        scaled[col] = scaled[col].apply(lambda v: custom_color(v, col))

    colorscale = [
        [0.0, "lightcoral"],   # red for bearish
        [0.5, "#eeeeee"],      # gray for neutral
        [1.0, "lightgreen"]    # green for bullish
    ]

    fig = px.imshow(
        scaled,
        color_continuous_scale=colorscale,
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)
