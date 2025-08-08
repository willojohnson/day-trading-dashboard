import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

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

# --- Tabs for navigation ---
tab1, tab2 = st.tabs(["Dashboard Overview", "Chart Analysis"])

# --- Sidebar Options ---
st.sidebar.header("Dashboard Settings")
refresh_rate = st.sidebar.slider("Refresh every N seconds", min_value=10, max_value=300, value=30, step=10)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# --- Interactive Strategy Selectors ---
all_bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Golden Cross", "Trend + MACD Bullish", "Ichimoku Bullish"]
all_bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Death Cross", "Death Cross + RSI Bearish", "Ichimoku Bearish"]
all_strategies = all_bullish_strategies + all_bearish_strategies

st.sidebar.markdown("### 🚦 Select Strategies")
selected_bullish = st.sidebar.multiselect("Bullish Strategies", all_bullish_strategies, default=all_bullish_strategies)
selected_bearish = st.sidebar.multiselect("Bearish Strategies", all_bearish_strategies, default=all_bearish_strategies)
selected_strategies = selected_bullish + selected_bearish

# --- Timeframe Selector ---
timeframe_options = ["5m", "15m", "30m", "1h", "1d", "3 month", "6 month", "YTD", "1 year", "5 year"]
timeframe = st.sidebar.selectbox("Select Timeframe", timeframe_options, index=4)

# --- Collapsible Strategy Definitions Section (Re-implemented) ---
with st.sidebar.expander("📘 Strategy Definitions", expanded=False):
    st.markdown("**Trend Trading**: 20MA > 50MA")
    st.markdown("**RSI Overbought**: RSI > 70")
    st.markdown("**RSI Oversold**: RSI < 30")
    st.markdown("**MACD Bullish Crossover**: MACD crosses above Signal")
    st.markdown("**MACD Bearish Crossover**: MACD crosses below Signal")
    st.markdown("**Death Cross**: 50MA crosses below 200MA")
    st.markdown("**Golden Cross**: 50MA crosses above 200MA")
    st.markdown("---")
    st.markdown("### Ichimoku Cloud (Ichimoku Kinko Hyo)")
    st.markdown("- **Tenkan-sen (Conversion Line)**: The average of the highest high and lowest low over the past 9 periods. Represents short-term momentum.")
    st.markdown("- **Kijun-sen (Base Line)**: The average of the highest high and lowest low over the past 26 periods. Represents long-term momentum.")
    st.markdown("- **Senkou Span A (Leading Span A)**: The average of the Tenkan-sen and Kijun-sen, plotted 26 periods ahead. One boundary of the Ichimoku cloud.")
    st.markdown("- **Senkou Span B (Leading Span B)**: The average of the highest high and lowest low over the past 52 periods, plotted 26 periods ahead. The other boundary of the Ichimoku cloud.")
    st.markdown("- **Chikou Span (Lagging Span)**: The current closing price, plotted 26 periods behind.")
    st.markdown("- **Ichimoku Bullish**: Price is above the cloud (Senkou Span A and B).")
    st.markdown("- **Ichimoku Bearish**: Price is below the cloud (Senkou Span A and B).")
    st.markdown("---")
    st.markdown("**Trend + MACD Bullish**: 20MA > 50MA AND MACD Bullish Crossover")
    st.markdown("**Death Cross + RSI Bearish**: 50MA < 200MA AND RSI > 70")


# --- Helper function for fetching and processing data ---
@st.cache_data(ttl=refresh_rate)
def fetch_and_process_data(ticker, timeframe):
    end_date = datetime.datetime.now()
    start_date = None
    interval = '1d'

    if timeframe in ['5m', '15m', '30m', '1h']:
        start_date = end_date - datetime.timedelta(days=7)
        interval = timeframe
    elif timeframe == '1d':
        start_date = end_date - datetime.timedelta(days=365)
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
    
    if start_date:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        st.error(f"Error: Unsupported timeframe '{timeframe}'.")
        return None, "Unsupported timeframe."

    if df.empty or 'Close' not in df.columns:
        return None, "No valid data."

    # Indicator Calculations
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()

    # RSI
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

# --- Main Logic ---
signals = []
heatmap_data = {strategy: np.zeros(len(TICKERS)) for strategy in all_strategies}
ticker_to_index = {ticker: i for i, ticker in enumerate(TICKERS)}

with st.spinner("⚙️ Processing data and generating signals..."):
    if selected_strategies:
        for ticker in TICKERS:
            company = TICKER_NAMES.get(ticker, ticker)
            df, error_msg = fetch_and_process_data(ticker, timeframe)

            if error_msg:
                st.warning(f"⚠️ {error_msg} for {ticker} ({company}). Skipping...")
                continue
            
            if len(df) < 200:
                st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators. Skipping strategy checks.")
                continue
            
            ma20_1, ma50_1, ma200_1, rsi_1, macd_1, macd_signal_1 = (
                df['20_MA'].iloc[-1].item(), df['50_MA'].iloc[-1].item(), df['200_MA'].iloc[-1].item(),
                df['RSI'].iloc[-1].item(), df['MACD'].iloc[-1].item(), df['MACD_Signal'].iloc[-1].item()
            )
            ma50_2, ma200_2, macd_2, macd_signal_2 = (
                df['50_MA'].iloc[-2].item(), df['200_MA'].iloc[-2].item(),
                df['MACD'].iloc[-2].item(), df['MACD_Signal'].iloc[-2].item()
            )

            macd_bullish_crossover = (macd_2 < macd_signal_2 and macd_1 > macd_signal_1)
            golden_cross = (ma50_2 < ma200_2 and ma50_1 > ma200_1)
            macd_bearish_crossover = (macd_2 > macd_signal_2 and macd_1 < macd_signal_1)
            death_cross = (ma50_2 > ma200_2 and ma50_1 < ma200_1)

            # Ichimoku
            last_close = df['Close'].iloc[-1].item()
            last_senkou_a = df['senkou_span_a'].iloc[-1].item()
            last_senkou_b = df['senkou_span_b'].iloc[-1].item()

            # Bullish Strategies
            ticker_idx = ticker_to_index[ticker]
            if "Trend Trading" in selected_bullish and ma20_1 > ma50_1:
                signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
                heatmap_data["Trend Trading"][ticker_idx] = 1
            if "RSI Oversold" in selected_bullish and rsi_1 < 30:
                signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={rsi_1:.1f})"))
                heatmap_data["RSI Oversold"][ticker_idx] = 1
            if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
                heatmap_data["MACD Bullish Crossover"][ticker_idx] = 1
            if "Golden Cross" in selected_bullish and golden_cross:
                signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
                heatmap_data["Golden Cross"][ticker_idx] = 1
            if "Trend + MACD Bullish" in selected_bullish and (ma20_1 > ma50_1) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_data["Trend + MACD Bullish"][ticker_idx] = 1
            if "Ichimoku Bullish" in selected_bullish and (last_close > last_senkou_a) and (last_close > last_senkou_b):
                 signals.append((ticker, "bullish", f"☁️ Bullish - Ichimoku Cloud Breakout — {company}"))
                 heatmap_data["Ichimoku Bullish"][ticker_idx] = 1
            
            # Bearish Strategies
            if "RSI Overbought" in selected_bearish and rsi_1 > 70:
                signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={rsi_1:.1f})"))
                heatmap_data["RSI Overbought"][ticker_idx] = -1
            if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
                signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
                heatmap_data["MACD Bearish Crossover"][ticker_idx] = -1
            if "Death Cross" in selected_bearish and death_cross:
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
                heatmap_data["Death Cross"][ticker_idx] = -1
            if "Death Cross + RSI Bearish" in selected_bearish and death_cross and (rsi_1 > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_data["Death Cross + RSI Bearish"][ticker_idx] = -1
            if "Ichimoku Bearish" in selected_bearish and (last_close < last_senkou_a) and (last_close < last_senkou_b):
                 signals.append((ticker, "bearish", f"☁️ Bearish - Ichimoku Cloud Breakdown — {company}"))
                 heatmap_data["Ichimoku Bearish"][ticker_idx] = -1
    else:
        st.info("⚠️ Please select at least one strategy from the sidebar to generate signals.")

# Create the heatmap DataFrame from the generated data
heatmap_df = pd.DataFrame(heatmap_data, index=TICKERS).T

# --- DASHBOARD OVERVIEW TAB ---
with tab1:
    # --- KPI Metrics at the Top ---
    st.markdown("### 📊 Market Overview")
    kpi_ticker = st.selectbox("Select a Ticker to view KPIs", TICKERS, index=0, key="kpi_select")
    
    kpi_df, _ = fetch_and_process_data(kpi_ticker, "1 year")
    
    if kpi_df is not None and not kpi_df.empty and len(kpi_df) >= 2:
        latest_price = kpi_df['Close'].iloc[-1].item()
        previous_price = kpi_df['Close'].iloc[-2].item()
        current_volume = kpi_df['Volume'].iloc[-1].item()
        low_price = kpi_df['Low'].iloc[-1].item()
        high_price = kpi_df['High'].iloc[-1].item()
        avg_volume = kpi_df['Volume'].mean().item()
        
        if not pd.isna(latest_price) and not pd.isna(previous_price):
            change_pct = ((latest_price - previous_price) / previous_price) * 100
            
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

    st.markdown("---")
    
    # --- Strategy Heatmap (Re-implemented) ---
    st.markdown("### 📊 Strategy Heatmap")
    if not heatmap_df.empty:
        # Create a Plotly figure for the heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale=[
                [0.0, 'rgb(255, 0, 0)'], # Red for -1 (bearish)
                [0.5, 'rgb(220, 220, 220)'], # Light gray for 0 (no signal)
                [1.0, 'rgb(0, 255, 0)'] # Green for 1 (bullish)
            ],
            colorbar=dict(
                tickvals=[-1, 0, 1],
                ticktext=['Bearish', 'No Signal', 'Bullish']
            )
        ))

        fig_heatmap.update_layout(
            title="Active Signals by Strategy and Ticker",
            xaxis_title="Tickers",
            yaxis_title="Strategies",
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # --- Signal Display ---
    st.markdown("### ✅ Current Trade Signals")
    if signals:
        for _, signal_type, msg in signals:
            if signal_type == "bullish":
                st.success(msg)
            elif signal_type == "bearish":
                st.error(msg)
    elif selected_strategies:
        st.info("No trade signals at this time for any active strategies.")
    else:
        st.info("Please select strategies from the sidebar to see current trade signals.")
    
    st.markdown("---")

# --- CHART ANALYSIS TAB ---
with tab2:
    st.markdown("### 🔎 Detailed Chart Analysis")
    chart_ticker = st.selectbox("Select a Ticker", options=TICKERS, key="chart_ticker")
    
    if chart_ticker:
        chart_df, error_msg = fetch_and_process_data(chart_ticker, timeframe)
        
        if chart_df is not None and not chart_df.empty:
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
                title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Price Action (Moving Averages)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_candlestick, use_container_width=True)

            fig_ichimoku = go.Figure(data=[
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['Open'],
                    high=chart_df['High'],
                    low=chart_df['Low'],
                    close=chart_df['Close'],
                    name='Price'
                ),
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

            fig_ichimoku.update_layout(
                title=f"{TICKER_NAMES.get(chart_ticker, chart_ticker)} Ichimoku Cloud ({timeframe} interval)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_ichimoku, use_container_width=True)

        else:
            st.warning(f"⚠️ No data available for {chart_ticker} at the selected timeframe.")
