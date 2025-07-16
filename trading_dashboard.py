import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh

# --- Tickers to Monitor ---
TICKERS = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SNOW", "AI", "AMD", "BBAI", "SOUN", "CRSP", "TSM", "DDOG"]

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Dashboard")

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=30, max_value=300, value=60, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

strategy = st.sidebar.selectbox("Select Strategy", ["Trend Trading", "RSI Overbought", "Scalping", "Breakout", "Lower High + Lower Low"])

# --- Strategy Definitions ---
st.sidebar.markdown("### 📘 Strategy Definitions")
st.sidebar.markdown("""
**Trend Trading**: Shows uptrend signals when 20MA > 50MA
**RSI Overbought**: Flags stocks with RSI > 70 for possible pullback
**Scalping**: Short-term trades triggered by volume surges and 20MA > 50MA
**Breakout**: Flags when current price breaks above 20-period high
**Lower High + Lower Low**: Detects weakening trends when each bar has a lower high and lower low than the previous bar
""")

# --- Data Processing & Signal Generation ---
now = datetime.datetime.now()
# Fetching data for the last 5 days with 5-minute interval might be too granular for some tickers or lead to missing data.
# Consider increasing the `days` value if you encounter frequent "No valid data" warnings.
start = now - datetime.timedelta(days=7) # Increased to 7 days for more data points
end = now

signals = []

st.subheader("⚙️ Processing Data and Generating Signals...") # Added a general processing message

for ticker in TICKERS:
    try:
        # yfinance interval "5m" typically requires data up to 60 days back for free tier,
        # but for real-time dashboard, a few days are usually sufficient.
        df = yf.download(ticker, start=start, end=end, interval="5m")

        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data for {ticker}. Skipping...")
            continue

        # Ensure enough data for calculations
        if len(df) < 50: # 50 is the max window size for 50MA
            st.info(f"ℹ️ Not enough data for {ticker} to calculate all indicators. Skipping...")
            continue

        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()

        # Corrected RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()

        # Handle division by zero for rs
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Add a small epsilon to avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
        # For breakout, we typically compare current price to the highest high of *previous* N periods.
        # So, calculate 20_High based on past 20 periods, excluding current bar.
        df['20_High'] = df['High'].rolling(window=20).max().shift(1)


        # --- Strategy Logic ---
        # Ensure that the last few data points are not NaN due to rolling window calculations
        current_close = df['Close'].iloc[-1]
        current_high = df['High'].iloc[-1]
        current_low = df['Low'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]

        ma20 = df['20_MA'].iloc[-1]
        ma50 = df['50_MA'].iloc[-1]
        rsi_val = df['RSI'].iloc[-1]
        avg_vol = df['Avg_Volume'].iloc[-1]
        high_20_period_prev = df['20_High'].iloc[-1] # This is the 20-period high up to the *previous* bar

        if pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi_val) or pd.isna(avg_vol) or pd.isna(high_20_period_prev):
             st.info(f"ℹ️ Not enough complete data for {ticker} to apply strategy. Skipping...")
             continue

        if strategy == "Trend Trading":
            if ma20 > ma50:
                signal = f"📈 **Trend Trading**: {ticker} - 20MA > 50MA"
                signals.append((ticker, signal))

        elif strategy == "RSI Overbought":
            if rsi_val > 70:
                signal = f"🔺 **RSI Overbought**: {ticker} - RSI={rsi_val:.1f}"
                signals.append((ticker, signal))

        elif strategy == "Scalping":
            if ma20 > ma50 and current_volume > 1.5 * avg_vol:
                signal = f"⚡ **Scalping**: {ticker} - Volume surge & 20MA > 50MA"
                signals.append((ticker, signal))

        elif strategy == "Breakout":
            # Check if current close breaks above the 20-period high (calculated from previous bars)
            if current_close > high_20_period_prev:
                signal = f"🔹 **Breakout**: {ticker} - Price > 20-period high"
                signals.append((ticker, signal))

        elif strategy == "Lower High + Lower Low":
            if len(df) >= 2: # Ensure at least two bars for comparison
                # Access previous bar's high and low directly
                prev_high = df['High'].iloc[-2]
                prev_low = df['Low'].iloc[-2]

                if (current_high < prev_high) and (current_low < prev_low):
                    signal = f"🔻 **Lower High + Lower Low**: {ticker} - Weakening trend"
                    signals.append((ticker, signal))

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {e}")

# --- Display Signals ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, msg in signals:
        st.success(msg)
else:
    st.info("No trade signals at this time for the selected strategy.")
