import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

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
# Updated list of options for the timeframe selectbox
timeframe_options = ["5m", "15m", "30m", "1h", "1d", "3 month", "6 month", "YTD", "1 year", "5 year"]
timeframe = st.sidebar.selectbox("Select Timeframe", timeframe_options, index=4)

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
    start_date = None
    interval = '1d' # Default interval for longer timeframes

    if timeframe == '5m' or timeframe == '15m' or timeframe == '30m' or timeframe == '1h':
        start_date = end_date - datetime.timedelta(days=7) # 7 days is a safe period for intraday intervals
        interval = timeframe
    elif timeframe == '1d':
        start_date = end_date - datetime.timedelta(days=365) # 1 year for daily
        interval = '1d'
    elif timeframe == '3 month':
        start_date = end_date - datetime.timedelta(days=90)
    elif timeframe == '6 month':
        start_date = end_date - datetime.timedelta(days=180)
    elif timeframe == 'YTD':
        start_date = datetime.datetime(end_date.year, 1, 1)
    elif timeframe == '1 year':
        start_date = end_date - datetime.timedelta(days=365)
    elif timeframe == '5 year':
        start_date = end_date - datetime.timedelta(days=5*365)
    
    # Check if start_date has been determined before downloading
    if start_date:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        # Fallback if an unexpected timeframe is selected
        st.error(f"Error: Unsupported timeframe '{timeframe}'.")
        return None, "Unsupported timeframe."

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
heatmap_matrix = pd.DataFrame(index=TICKERS, columns=all_bullish_strategies + all_bearish_strategies).fillna('')

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

        # Reusable conditions
        macd_bullish_crossover = (macd_2 < macd_signal_2 and macd_1 > macd_signal_1)
        golden_cross = (ma50_2 < ma200_2 and ma50_1 > ma200_1)
        macd_bearish_crossover = (macd_2 > macd_signal_2 and macd_1 < macd_signal_1)
        death_cross = (ma50_2 > ma200_2 and ma50_1 < ma200_1)

        # Bullish Strategies
        if "Trend Trading" in selected_bullish and ma20_1 > ma50_1:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_matrix.loc[ticker, "Trend Trading"] = "✔"

        if "RSI Oversold" in selected_bullish and rsi_1 < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={rsi_1:.1f})"))
            heatmap_matrix.loc[ticker, "RSI Oversold"] = f"{rsi_1:.1f}"

        if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
            signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
            heatmap_matrix.loc[ticker, "MACD Bullish Crossover"] = "✔"
        
        if "Golden Cross" in selected_bullish and golden_cross:
            signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
            heatmap_matrix.loc[ticker, "Golden Cross"] = "✔"
            st.success(f"🔥 GOLDEN CROSS DETECTED for **{ticker} - {company}**!") # <<< FLASH NOTIFICATION
        
        if "Trend + MACD Bullish" in selected_bullish:
            if (ma20_1 > ma50_1) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_matrix.loc[ticker, "Trend + MACD Bullish"] = "✔"

        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and rsi_1 > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={rsi_1:.1f})"))
            heatmap_matrix.loc[ticker, "RSI Overbought"] = f"{rsi_1:.1f}"

        if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
            signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
            heatmap_matrix.loc[ticker, "MACD Bearish Crossover"] = "✔"
        
        if "Death Cross" in selected_bearish and death_cross:
            signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
            heatmap_matrix.loc[ticker, "Death Cross"] = "✔"
            st.error(f"🚨 DEATH CROSS DETECTED for **{ticker} - {company}**!") # <<< FLASH NOTIFICATION
        
        if "Death Cross + RSI Bearish" in selected_bearish:
            if death_cross and (rsi_1 > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_matrix.loc[ticker, "Death Cross + RSI Bearish"] = "✔"


# --- DASHBOARD OVERVIEW TAB ---
with tab1:
    # --- KPI Metrics at the Top ---
    st.markdown("### 📊 Market Overview")
    kpi_ticker = st.selectbox("Select a Ticker to view KPIs", TICKERS, index=0, key="kpi_select")
    
    # We always need at least a year of daily data for the KPIs, so we force that
    kpi_df, _ = fetch_and_process_data(kpi_ticker, "1 year")
    
    # Updated logic for KPI section to prevent errors on insufficient data
    if kpi_df is not None and not kpi_df.empty and len(kpi_df) >= 2:
        # Explicitly get the scalar value using .item() to avoid ValueError
        latest_price = kpi_df['Close'].iloc[-1].item()
        previous_price = kpi_df['Close'].iloc[-2].item()
        current_volume = kpi_df['Volume'].iloc[-1].item()
        low_price = kpi_df['Low'].iloc[-1].item()
        high_price = kpi_df['High'].iloc[-1].item()
        avg_volume = kpi_df['Volume'].mean().item()
        
        if not pd.isna(latest_price) and not pd.isna(previous_price):
            change_pct = ((latest_price - previous_price) / previous_price) * 100
            
            # Reordered columns to group related metrics
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
    if not heatmap_matrix.empty:
        st.markdown("### 🧭 Strategy Signal Matrix")
        
        # Create a display DataFrame with full company names for the `st.dataframe` table
        display_matrix = heatmap_matrix.copy()
        display_matrix.index = [f"{ticker} — {TICKER_NAMES.get(ticker, ticker)}" for ticker in display_matrix.index]
        
        def to_numeric_signal(val):
            return 1 if val != "" else 0

        numeric_heatmap_df = display_matrix[all_bullish_strategies + all_bearish_strategies].applymap(to_numeric_signal)
        display_matrix["Bullish Total"] = numeric_heatmap_df[all_bullish_strategies].sum(axis=1)
        display_matrix["Bearish Total"] = numeric_heatmap_df[all_bearish_strategies].sum(axis=1)

        ordered_cols = all_bullish_strategies + ["Bullish Total"] + all_bearish_strategies + ["Bearish Total"]
        display_matrix = display_matrix[ordered_cols]

        # --- NEW Style function to highlight both non-zero totals ---
        def highlight_total_signals_v2(row):
            styles = [''] * len(row)
            bullish_total = row["Bullish Total"]
            bearish_total = row["Bearish Total"]
            
            bullish_total_idx = len(all_bullish_strategies)
            bearish_total_idx = len(all_bullish_strategies) + len(all_bearish_strategies) + 1
            
            # Apply green highlight if bullish total is greater than 0
            if bullish_total > 0:
                styles[bullish_total_idx] = 'background-color: #d4edda; color: #155724;' # Light green
            
            # Apply red highlight if bearish total is greater than 0
            if bearish_total > 0:
                styles[bearish_total_idx] = 'background-color: #f8d7da; color: #721c24;' # Light red
            
            return styles
            
        st.dataframe(
            display_matrix.style
            .apply(highlight_total_signals_v2, axis=1) # Use the new function
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6')]},
                {'selector': '.st-table', 'props': [('width', '100%')]}
            ]),
            use_container_width=True
        )

        st.markdown("### 🔥 Strategy Activation Heatmap")

        # Create a numerical matrix for coloring
        color_data = []
        for index, row in heatmap_matrix.iterrows():
            row_colors = []
            for col in heatmap_matrix.columns:
                val = row[col]
                if val != "":
                    if col in all_bullish_strategies:
                        row_colors.append(1) # Bullish -> Green
                    elif col in all_bearish_strategies:
                        row_colors.append(-1) # Bearish -> Red
                    else:
                        row_colors.append(0) # Should not happen, but for safety
                else:
                    row_colors.append(0) # No signal -> Neutral
            color_data.append(row_colors)

        # Create a single Plotly figure with a heatmap
        fig = go.Figure(go.Heatmap(
            z=color_data,
            x=heatmap_matrix.columns,
            y=[f"{ticker} — {TICKER_NAMES.get(ticker, ticker)}" for ticker in heatmap_matrix.index],
            text=heatmap_matrix.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale=[[0, 'lightcoral'], [0.5, 'white'], [1, 'lightgreen']],
            zmin=-1, zmax=1,
            xgap=1, ygap=1
        ))

        # This line forces the y-axis to match the top-to-bottom order of the table
        fig.update_layout(
            title="Strategy Activation Heatmap",
            xaxis_title="Strategy",
            yaxis_title="Ticker",
            yaxis={'autorange': 'reversed'},
            xaxis={'side': 'top'},
            margin=dict(t=50, b=50, l=50, r=50),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# --- CHART ANALYSIS TAB ---
with tab2:
    st.markdown("### 🔎 Detailed Chart Analysis")
    chart_ticker = st.selectbox("Select a Ticker", options=TICKERS, key="chart_ticker")
    
    if chart_ticker:
        chart_df, error_msg = fetch_and_process_data(chart_ticker, timeframe)
        
        if chart_df is not None and not chart_df.empty:
            # Create candlestick chart
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
                title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Price Action ({timeframe} interval)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_candlestick, use_container_width=True)

        else:
            st.warning(f"⚠️ No data available for {chart_ticker} at the selected timeframe.")
