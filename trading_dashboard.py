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
bullish_strategies = [
    "Trend Trading", "MACD Bullish Crossover", "RSI Oversold",
    "Golden Cross", "Trend + MACD Bullish", "Golden Cross + Volume",
    "Bollinger Bullish Breakout",
    "OBV Bullish Trend Confirmation" 
]
bearish_strategies = [
    "MACD Bearish Crossover", "RSI Overbought", "Death Cross",
    "Death Cross + RSI Bearish", "Death Cross + Volume",
    "Bollinger Bearish Breakout",
    "OBV Bearish Trend Confirmation" 
]

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
st.sidebar.markdown("**Golden Cross + Volume**: Golden Cross AND Above-Avg Volume (1.2x 20-period Avg Volume)")
st.sidebar.markdown("**Death Cross + Volume**: Death Cross AND Above-Avg Volume (1.2x 20-period Avg Volume)")
st.sidebar.markdown("**Bollinger Bullish Breakout**: Close crosses above Upper Bollinger Band (20-period)")
st.sidebar.markdown("**Bollinger Bearish Breakout**: Close crosses below Lower Bollinger Band (20-period)")
st.sidebar.markdown("---")
st.sidebar.markdown("**OBV Bullish Trend Confirmation**: Current Close > Previous Close AND Current OBV > Previous OBV") 
st.sidebar.markdown("**OBV Bearish Trend Confirmation**: Current Close < Previous Close AND Current OBV < Previous OBV") 


# --- Signal Detection ---
now = datetime.datetime.now()
start = now - datetime.timedelta(days=60) # Max for 5m interval is typically 60 days.
end = now

signals = []
heatmap_data = []

st.subheader("⚙️ Processing Data and Generating Signals...")

for ticker in TICKERS:
    company = TICKER_NAMES.get(ticker, ticker)
    try:
        df = yf.download(ticker, start=start, end=end, interval="5m")

        if df.empty or 'Close' not in df.columns:
            st.warning(f"⚠️ No valid data or 'Close' column for {ticker} ({company}). Skipping...")
            continue

        if len(df) < 200: # Need at least 200 periods for 200MA
            st.info(f"ℹ️ Not enough historical data for {ticker} ({company}) for 200-period MA or Bollinger Band calculation (need 200 bars). Skipping strategies.")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = 0
            heatmap_data.append(heatmap_row)
            continue
        
        if len(df) < 2: # Ensure at least 2 bars for any crossover logic that looks at iloc[-2]
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) for crossover analysis (requires at least 2 bars). Skipping strategy checks.")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = 0
            heatmap_data.append(heatmap_row)
            continue


        # --- Indicator Calculations ---
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

        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = df['20_MA'] + (df['StdDev'] * 2)
        df['Lower_BB'] = df['20_MA'] - (df['StdDev'] * 2)

        # On-Balance Volume (OBV)
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]


        # Signal Matrix Row Initialization
        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = 0

        # --- CRUCIAL: Extract scalar values using .item() and handle NaNs ---
        def get_scalar_value(series_or_scalar):
            if pd.isna(series_or_scalar):
                return None
            try:
                # Attempt to get the scalar value using .item()
                return series_or_scalar.item()
            except AttributeError:
                # If it doesn't have .item(), it might already be a scalar (int, float, etc.)
                # We need to ensure it's a true scalar.
                if isinstance(series_or_scalar, (int, float, bool)):
                    return series_or_scalar
                # If it's not a basic scalar type and has no .item(), it's problematic
                # Treat as None, as it's not a clean scalar for comparisons.
                return None
            except Exception as e:
                # Catch any other unexpected errors during item extraction
                print(f"DEBUG: Failed to extract scalar for {type(series_or_scalar)} (value: {series_or_scalar}). Error: {e}")
                return None

        # Initialize all variables to None
        ma20_1, ma50_1, ma200_1, rsi_1, macd_1, macd_signal_1, volume_1, avg_volume_1 = [None] * 8
        ma50_2, ma200_2, macd_2, macd_signal_2 = [None] * 4
        close_1, close_2, upper_bb_1, upper_bb_2, lower_bb_1, lower_bb_2 = [None] * 6
        
        # OBV scalars
        obv_1, obv_2 = [None] * 2

        try:
            # Current values (iloc[-1])
            if not df.empty and len(df) >= 1:
                ma20_1 = get_scalar_value(df['20_MA'].iloc[-1])
                ma50_1 = get_scalar_value(df['50_MA'].iloc[-1])
                ma200_1 = get_scalar_value(df['200_MA'].iloc[-1])
                rsi_1 = get_scalar_value(df['RSI'].iloc[-1])
                macd_1 = get_scalar_value(df['MACD'].iloc[-1])
                macd_signal_1 = get_scalar_value(df['MACD_Signal'].iloc[-1])
                volume_1 = get_scalar_value(df['Volume'].iloc[-1])
                avg_volume_1 = get_scalar_value(df['Avg_Volume'].iloc[-1])
                close_1 = get_scalar_value(df['Close'].iloc[-1])
                upper_bb_1 = get_scalar_value(df['Upper_BB'].iloc[-1])
                lower_bb_1 = get_scalar_value(df['Lower_BB'].iloc[-1])
                obv_1 = get_scalar_value(df['OBV'].iloc[-1])

            # Previous values (iloc[-2])
            if not df.empty and len(df) >= 2:
                macd_2 = get_scalar_value(df['MACD'].iloc[-2])
                macd_signal_2 = get_scalar_value(df['MACD_Signal'].iloc[-2])
                ma50_2 = get_scalar_value(df['50_MA'].iloc[-2])
                ma200_2 = get_scalar_value(df['200_MA'].iloc[-2])
                close_2 = get_scalar_value(df['Close'].iloc[-2])
                upper_bb_2 = get_scalar_value(df['Upper_BB'].iloc[-2])
                lower_bb_2 = get_scalar_value(df['Lower_BB'].iloc[-2])
                obv_2 = get_scalar_value(df['OBV'].iloc[-2])

        except ValueError as ve:
            st.error(f"❌ Value Error during scalar extraction for {ticker} ({company}): {ve}. This might indicate unexpected data. Skipping strategy checks.")
            print(f"DEBUG: Scalar extraction ValueError for {ticker}: {ve}")
            heatmap_data.append(heatmap_row)
            continue
        except IndexError as ie:
            st.error(f"❌ Index Error during scalar extraction for {ticker} ({company}): {ie}. Not enough data points to extract latest values. Skipping strategy checks.")
            heatmap_data.append(heatmap_row)
            continue
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during scalar extraction for {ticker} ({company}): {e}. Skipping strategy checks.")
            print(f"DEBUG: Unhandled error during scalar extraction for {ticker}: {e}")
            heatmap_data.append(heatmap_row)
            continue

        # Consolidated check for sufficient & valid data for strategy evaluation
        required_scalars = [ma20_1, ma50_1, ma200_1, rsi_1, macd_1, macd_signal_1,
                            volume_1, avg_volume_1,
                            macd_2, macd_signal_2, ma50_2, ma200_2,
                            close_1, close_2, upper_bb_1, upper_bb_2, lower_bb_1, lower_bb_2,
                            obv_1, obv_2] 

        if any(s is None for s in required_scalars):
            st.info(f"ℹ️ Not enough complete indicator data (or data conversion issues) for {ticker} ({company}). Skipping strategy checks for this ticker.")
            for s_name, s_value in zip(['ma20_1', 'ma50_1', 'ma200_1', 'rsi_1', 'macd_1', 'macd_signal_1',
                                        'volume_1', 'avg_volume_1', 'macd_2', 'macd_signal_2', 'ma50_2', 'ma200_2',
                                        'close_1', 'close_2', 'upper_bb_1', 'upper_bb_2', 'lower_bb_1', 'lower_bb_2',
                                        'obv_1', 'obv_2'], required_scalars):
                if s_value is None:
                    print(f"DEBUG: For {ticker}, '{s_name}' is None.")
            heatmap_data.append(heatmap_row)
            continue
        
        # --- Reusable conditions using the extracted scalar variables ---
        is_above_avg_volume = False
        if avg_volume_1 != 0:
            is_above_avg_volume = volume_1 > 1.2 * avg_volume_1

        macd_bullish_crossover = (macd_2 < macd_signal_2 and macd_1 > macd_signal_1)
        golden_cross = (ma50_2 < ma200_2 and ma50_1 > ma200_1)
        macd_bearish_crossover = (macd_2 > macd_signal_2 and macd_1 < macd_signal_1)
        death_cross = (ma50_2 > ma200_2 and ma50_1 < ma200_1)

        bollinger_bullish_breakout = (close_2 <= upper_bb_2 and close_1 > upper_bb_1)
        bollinger_bearish_breakout = (close_2 >= lower_bb_2 and close_1 < lower_bb_1)

        # OBV Trend Confirmation
        obv_bullish_trend_confirm = (close_1 > close_2 and obv_1 > obv_2)
        obv_bearish_trend_confirm = (close_1 < close_2 and obv_1 < obv_2)

        # Bullish Strategies
        if "Trend Trading" in selected_bullish and ma20_1 > ma50_1:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = 1

        if "RSI Oversold" in selected_bullish and rsi_1 < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Oversold"] = 1

        if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
            signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
            heatmap_row["MACD Bullish Crossover"] = 1
        
        if "Golden Cross" in selected_bullish and golden_cross:
            signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
            heatmap_row["Golden Cross"] = 1

        # --- Bullish Confirmation Strategies ---
        if "Trend + MACD Bullish" in selected_bullish:
            if (ma20_1 > ma50_1) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_row["Trend + MACD Bullish"] = 1
        
        if "Golden Cross + Volume" in selected_bullish:
            if golden_cross and is_above_avg_volume:
                signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross + Volume Confirmed — {company}"))
                heatmap_row["Golden Cross + Volume"] = 1
        
        if "Bollinger Bullish Breakout" in selected_bullish and bollinger_bullish_breakout:
            signals.append((ticker, "bullish", f"💥 Bullish - Bollinger Breakout — {company}"))
            heatmap_row["Bollinger Bullish Breakout"] = 1
        
        # OBV Bullish Strategies
        if "OBV Bullish Trend Confirmation" in selected_bullish and obv_bullish_trend_confirm:
            signals.append((ticker, "bullish", f"📊 Bullish - OBV Trend Confirmed — {company}"))
            heatmap_row["OBV Bullish Trend Confirmation"] = 1


        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and rsi_1 > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Overbought"] = 1

        if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
            signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
            heatmap_row["MACD Bearish Crossover"] = 1
        
        if "Death Cross" in selected_bearish and death_cross:
            signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
            heatmap_row["Death Cross"] = 1
        
        # --- Bearish Confirmation Strategies ---
        if "Death Cross + RSI Bearish" in selected_bearish:
            if death_cross and (rsi_1 > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_row["Death Cross + RSI Bearish"] = 1

        if "Death Cross + Volume" in selected_bearish:
            if death_cross and is_above_avg_volume:
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + Volume Confirmed — {company}"))
                heatmap_row["Death Cross + Volume"] = 1

        if "Bollinger Bearish Breakout" in selected_bearish and bollinger_bearish_breakout:
            signals.append((ticker, "bearish", f"📉 Bearish - Bollinger Breakout — {company}"))
            heatmap_row["Bollinger Bearish Breakout"] = 1

        # OBV Bearish Strategies
        if "OBV Bearish Trend Confirmation" in selected_bearish and obv_bearish_trend_confirm:
            signals.append((ticker, "bearish", f"📉 Bearish - OBV Trend Confirmed — {company}"))
            heatmap_row["OBV Bearish Trend Confirmation"] = 1

        heatmap_data.append(heatmap_row)

    except Exception as e:
        st.error(f"❌ An unexpected error occurred while processing {ticker} ({company}): {e}")
        print(f"DEBUG: Unhandled error for {ticker}: {e}") # This will print to your terminal

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

    # --- CUSTOM STYLING FOR BEARISH TOTAL ---
    def highlight_bearish_total(val):
        """Highlights the 'Bearish Total' cell red if value is 1 or more, otherwise no color."""
        if val >= 1:
            return 'background-color: salmon'
        return '' # No background color

    st.dataframe(
        heatmap_df.style
        .highlight_max(axis=0, subset=["Bullish Total"], color="lightgreen")
        .applymap(highlight_bearish_total, subset=['Bearish Total']) # Apply custom styling
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
