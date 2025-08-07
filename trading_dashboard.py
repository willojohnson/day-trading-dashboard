import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go # New import for candlestick chart

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
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=10, max_value=300, value=30, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Interactive Strategy Selectors ---
all_bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Golden Cross", "Trend + MACD Bullish"]
all_bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Death Cross", "Death Cross + RSI Bearish"]

st.sidebar.markdown("### 🚦 Select Strategies")
selected_bullish = st.sidebar.multiselect("Bullish Strategies", all_bullish_strategies, default=all_bullish_strategies)
selected_bearish = st.sidebar.multiselect("Bearish Strategies", all_bearish_strategies, default=all_bearish_strategies)

# --- Timeframe Selector ---
timeframe = st.sidebar.selectbox("Select Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=0)

# --- Strategy Definitions (in a collapsible expander) ---
with st.sidebar.expander("📘 Strategy Definitions"):
    st.markdown("**Trend Trading**: 20MA > 50MA")
    st.markdown("**RSI Overbought**: RSI > 70")
    st.markdown("**RSI Oversold**: RSI < 30")
    st.markdown("**MACD Bullish Crossover**: MACD crosses above Signal")
    st.markdown("**MACD Bearish Crossover**: MACD crosses below Signal")
    st.markdown("**Death Cross**: 50MA crosses below 200MA")
    st.markdown("**Golden Cross**: 50MA crosses above 200MA")
    st.markdown("---")
    st.markdown("**Trend + MACD Bullish**: 20MA > 50MA AND MACD Bullish Crossover")
    st.markdown("**Death Cross + RSI Bearish**: 50MA < 200MA AND RSI > 70")


# --- Tabs for Content Organization ---
tab1, tab2 = st.tabs(["📊 Dashboard Overview", "📈 Chart Analysis"])

# --- Helper function for fetching and processing data ---
@st.cache_data(ttl=refresh_rate)
def fetch_and_process_data(ticker, timeframe):
    end_date = datetime.datetime.now()
    if timeframe == '1d':
        start_date = end_date - datetime.timedelta(days=365) # 1 year for daily
    elif timeframe == '1h':
        start_date = end_date - datetime.timedelta(days=60) # 60 days for hourly
    else: # 5m, 15m, 30m intervals
        start_date = end_date - datetime.timedelta(days=7) # 7 days is a safe period
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)
    if df.empty or 'Close' not in df.columns:
        return None, "No valid data."

    # Indicator Calculations
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()

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

    return df, None

# --- Main Logic ---
signals = []
heatmap_data = []

with st.spinner("⚙️ Processing data and generating signals..."):
    for ticker in TICKERS:
        company = TICKER_NAMES.get(ticker, ticker)
        df, error_msg = fetch_and_process_data(ticker, timeframe)

        if error_msg:
            st.warning(f"⚠️ {error_msg} for {ticker} ({company}). Skipping...")
            continue
        
        # Check if we have enough data for all indicators
        if len(df) < 200:
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators. Skipping strategy checks.")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in all_bullish_strategies + all_bearish_strategies:
                heatmap_row[strat] = ""
            heatmap_data.append(heatmap_row)
            continue
        
        # Extract scalar values from the last two rows
        ma20_1, ma50_1, ma200_1, rsi_1, macd_1, macd_signal_1 = (
            df['20_MA'].iloc[-1], df['50_MA'].iloc[-1], df['200_MA'].iloc[-1],
            df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['MACD_Signal'].iloc[-1]
        )
        ma50_2, ma200_2, macd_2, macd_signal_2 = (
            df['50_MA'].iloc[-2], df['200_MA'].iloc[-2],
            df['MACD'].iloc[-2], df['MACD_Signal'].iloc[-2]
        )

        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in all_bullish_strategies + all_bearish_strategies:
            heatmap_row[strat] = ""

        # Reusable conditions
        macd_bullish_crossover = (macd_2 < macd_signal_2 and macd_1 > macd_signal_1)
        golden_cross = (ma50_2 < ma200_2 and ma50_1 > ma200_1)
        macd_bearish_crossover = (macd_2 > macd_signal_2 and macd_1 < macd_signal_1)
        death_cross = (ma50_2 > ma200_2 and ma50_1 < ma200_1)

        # Bullish Strategies
        if "Trend Trading" in selected_bullish and ma20_1 > ma50_1:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = "✔"

        if "RSI Oversold" in selected_bullish and rsi_1 < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Oversold"] = f"{rsi_1:.1f}"

        if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
            signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
            heatmap_row["MACD Bullish Crossover"] = "✔"
        
        if "Golden Cross" in selected_bullish and golden_cross:
            signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
            heatmap_row["Golden Cross"] = "✔"
        
        if "Trend + MACD Bullish" in selected_bullish:
            if (ma20_1 > ma50_1) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_row["Trend + MACD Bullish"] = "✔"

        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and rsi_1 > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Overbought"] = f"{rsi_1:.1f}"

        if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
            signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
            heatmap_row["MACD Bearish Crossover"] = "✔"
        
        if "Death Cross" in selected_bearish and death_cross:
            signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
            heatmap_row["Death Cross"] = "✔"
        
        if "Death Cross + RSI Bearish" in selected_bearish:
            if death_cross and (rsi_1 > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_row["Death Cross + RSI Bearish"] = "✔"
        
        heatmap_data.append(heatmap_row)


# --- DASHBOARD OVERVIEW TAB ---
with tab1:
    # --- KPI Metrics at the Top ---
    st.markdown("### 📊 Market Overview")
    kpi_ticker = st.selectbox("Select a Ticker to view KPIs", TICKERS, index=0, key="kpi_select")
    
    kpi_df, _ = fetch_and_process_data(kpi_ticker, "1d")
    
    # Updated logic for KPI section to prevent errors on insufficient data
    if kpi_df is not None and not kpi_df.empty and len(kpi_df) >= 2:
        # Explicitly get the scalar value using .item() to avoid ValueError
        latest_price = kpi_df['Close'].iloc[-1].item()
        previous_price = kpi_df['Close'].iloc[-2].item()
        volume = kpi_df['Volume'].iloc[-1].item()
        low_price = kpi_df['Low'].iloc[-1].item()
        high_price = kpi_df['High'].iloc[-1].item()
        
        if not pd.isna(latest_price) and not pd.isna(previous_price):
            change_pct = ((latest_price - previous_price) / previous_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label=f"Price ({kpi_ticker})", value=f"${latest_price:.2f}", delta=f"{change_pct:.2f}%")
            with col2:
                st.metric(label="Volume", value=f"{volume:,}")
            with col3:
                # Use the new scalar variables here
                st.metric(label="Today's Range", value=f"${low_price:.2f} - ${high_price:.2f}")
        else:
            st.warning(f"⚠️ No recent pricing data for {kpi_ticker} to calculate KPIs. Please check back later.")
    else:
        st.warning(f"⚠️ Insufficient data for {kpi_ticker} to calculate KPIs. Please check back later.")


    # --- Signal Display ---
    st.markdown("### ✅ Current Trade Signals")
    if signals:
        for _, signal_type, msg in signals:
            if signal_type == "bullish":
                st.success(msg)
            elif signal_type == "bearish":
                st.error(msg)
    else:
        st.info("No trade signals at this time for any active strategies.")
    
    st.markdown("---")
    
    # --- Heatmap Matrix + Visual ---
    if heatmap_data:
        st.markdown("### 🧭 Strategy Signal Matrix")
        
        heatmap_df = pd.DataFrame(heatmap_data)
        for strat in all_bullish_strategies + all_bearish_strategies:
            if strat not in heatmap_df.columns:
                heatmap_df[strat] = ""

        def to_numeric_signal(val):
            return 1 if val != "" else 0

        numeric_heatmap_df = heatmap_df[all_bullish_strategies + all_bearish_strategies].applymap(to_numeric_signal)
        heatmap_df["Bullish Total"] = numeric_heatmap_df[all_bullish_strategies].sum(axis=1)
        heatmap_df["Bearish Total"] = numeric_heatmap_df[all_bearish_strategies].sum(axis=1)

        ordered_cols = ["Label"] + all_bullish_strategies + ["Bullish Total"] + all_bearish_strategies + ["Bearish Total"]
        heatmap_df = heatmap_df[ordered_cols]

        def highlight_total_signals(row):
            styles = [''] * len(row)
            bullish_total = row["Bullish Total"]
            bearish_total = row["Bearish Total"]
            
            if bullish_total > 0 and bullish_total > bearish_total:
                styles[-2] = 'background-color: #d4edda; color: #155724;' # Light green for bullish
            elif bearish_total > 0 and bearish_total > bullish_total:
                styles[-1] = 'background-color: #f8d7da; color: #721c24;' # Light red for bearish
            return styles
            
        st.dataframe(
            heatmap_df.style
            .apply(highlight_total_signals, axis=1)
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                {'selector': '.st-table', 'props': [('width', '100%')]}
            ]),
            use_container_width=True
        )

        st.markdown("### 🔥 Strategy Activation Heatmap")

        matrix = heatmap_df.set_index("Label")[all_bullish_strategies + all_bearish_strategies]
        
        # New, more explicit coloring logic for the heatmap
        color_matrix = matrix.copy().astype(str)
        for col in color_matrix.columns:
            if col in all_bullish_strategies:
                color_matrix[col] = color_matrix[col].apply(lambda x: 'lightgreen' if x != "" else 'transparent')
            elif col in all_bearish_strategies:
                color_matrix[col] = color_matrix[col].apply(lambda x: 'lightcoral' if x != "" else 'transparent')

        fig = go.Figure(data=go.Heatmap(
            z=matrix.where(matrix != "", 0).astype(bool).astype(int),
            x=matrix.columns,
            y=matrix.index,
            colorscale=[[0, 'white'], [1, 'lightgreen']],
            showscale=False,
            text=matrix.values,
            texttemplate="%{text}",
            xgap=3, ygap=3,
        ))

        for col in all_bearish_strategies:
            bearish_mask = (matrix[col] != "")
            if bearish_mask.any():
                fig.add_trace(go.Heatmap(
                    z=matrix[col].where(matrix[col] == "", 0).astype(bool).astype(int),
                    x=[col],
                    y=matrix.index[bearish_mask],
                    colorscale=[[0, 'white'], [1, 'lightcoral']],
                    showscale=False,
                    text=matrix[col].loc[bearish_mask],
                    texttemplate="%{text}",
                    xgap=3, ygap=3,
                    opacity=1
                ))
        
        fig.update_layout(
            title="Strategy Activation Heatmap",
            xaxis_title="Strategy",
            yaxis_title="Ticker",
            legend_title="Color Scale",
            autosize=True,
            margin=dict(t=30, b=30, l=30, r=30),
            xaxis={'side': 'top'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
