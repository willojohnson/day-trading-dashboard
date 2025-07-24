import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
bullish_strategies = ["Trend Trading", "MACD Bullish Crossover", "RSI Oversold", "Golden Cross", "Trend + MACD Bullish"]
bearish_strategies = ["MACD Bearish Crossover", "RSI Overbought", "Death Cross", "Death Cross + RSI Bearish"]

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


# --- Signal Detection (Existing Logic) ---
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
            st.warning(f"⚠️ No valid data for {ticker} ({company}). Skipping...")
            continue

        if len(df) < 200: # Need at least 200 periods for 200MA
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) to calculate all indicators (requires 200 bars). Skipping...")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = "" # Initialize with empty string for no signal
            heatmap_data.append(heatmap_row)
            continue
        
        if len(df) < 2:
            st.info(f"ℹ️ Not enough data for {ticker} ({company}) for crossover analysis (requires at least 2 bars). Skipping strategy checks.")
            heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
            for strat in bullish_strategies + bearish_strategies:
                heatmap_row[strat] = "" # Initialize with empty string for no signal
            heatmap_data.append(heatmap_row)
            continue


        # --- Indicator Calculations ---
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


        # Signal Matrix Row Initialization
        heatmap_row = {"Ticker": ticker, "Label": f"{ticker} — {company}"}
        for strat in bullish_strategies + bearish_strategies:
            heatmap_row[strat] = "" # Initialize with empty string for no signal

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
        ma20_1, ma50_1, ma200_1, rsi_1, macd_1, macd_signal_1 = [None] * 6
        ma50_2, ma200_2, macd_2, macd_signal_2 = [None] * 4

        try:
            # Current values (iloc[-1])
            if not df.empty and len(df) >= 1:
                ma20_1 = get_scalar_value(df['20_MA'].iloc[-1])
                ma50_1 = get_scalar_value(df['50_MA'].iloc[-1])
                ma200_1 = get_scalar_value(df['200_MA'].iloc[-1])
                rsi_1 = get_scalar_value(df['RSI'].iloc[-1])
                macd_1 = get_scalar_value(df['MACD'].iloc[-1])
                macd_signal_1 = get_scalar_value(df['MACD_Signal'].iloc[-1])

            # Previous values (iloc[-2])
            if not df.empty and len(df) >= 2:
                macd_2 = get_scalar_value(df['MACD'].iloc[-2])
                macd_signal_2 = get_scalar_value(df['MACD_Signal'].iloc[-2])
                ma50_2 = get_scalar_value(df['50_MA'].iloc[-2])
                ma200_2 = get_scalar_value(df['200_MA'].iloc[-2])

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
                            macd_2, macd_signal_2, ma50_2, ma200_2]

        if any(s is None for s in required_scalars):
            st.info(f"ℹ️ Not enough complete indicator data (or data conversion issues) for {ticker} ({company}). Skipping strategy checks for this ticker.")
            for s_name, s_value in zip(['ma20_1', 'ma50_1', 'ma200_1', 'rsi_1', 'macd_1', 'macd_signal_1',
                                        'macd_2', 'macd_signal_2', 'ma50_2', 'ma200_2'], required_scalars):
                if s_value is None:
                    print(f"DEBUG: For {ticker}, '{s_name}' is None.")
            heatmap_data.append(heatmap_row)
            continue

        # --- Reusable conditions using the extracted scalar variables ---
        macd_bullish_crossover = (macd_2 < macd_signal_2 and macd_1 > macd_signal_1)
        golden_cross = (ma50_2 < ma200_2 and ma50_1 > ma200_1)
        macd_bearish_crossover = (macd_2 > macd_signal_2 and macd_1 < macd_signal_1)
        death_cross = (ma50_2 > ma200_2 and ma50_1 < ma200_1)


        # Bullish Strategies
        if "Trend Trading" in selected_bullish and ma20_1 > ma50_1:
            signals.append((ticker, "bullish", f"📈 Bullish - Trend Trading — {company}"))
            heatmap_row["Trend Trading"] = "✔" # Checkmark for signal

        if "RSI Oversold" in selected_bullish and rsi_1 < 30:
            signals.append((ticker, "bullish", f"📈 Bullish - RSI Oversold — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Oversold"] = f"{rsi_1:.2f}" # Store exact RSI value

        if "MACD Bullish Crossover" in selected_bullish and macd_bullish_crossover:
            signals.append((ticker, "bullish", f"📈 Bullish - MACD Bullish Crossover — {company}"))
            heatmap_row["MACD Bullish Crossover"] = "✔" # Checkmark for signal
        
        if "Golden Cross" in selected_bullish and golden_cross:
            signals.append((ticker, "bullish", f"✨ Bullish - Golden Cross — {company}"))
            heatmap_row["Golden Cross"] = "✔" # Checkmark for signal

        # --- New Bullish Confirmation Strategy ---
        if "Trend + MACD Bullish" in selected_bullish:
            if (ma20_1 > ma50_1) and macd_bullish_crossover:
                signals.append((ticker, "bullish", f"✨ Bullish - Trend + MACD Confirmed — {company}"))
                heatmap_row["Trend + MACD Bullish"] = "✔" # Checkmark for signal


        # Bearish Strategies
        if "RSI Overbought" in selected_bearish and rsi_1 > 70:
            signals.append((ticker, "bearish", f"📉 Bearish - RSI Overbought — {company} (RSI={rsi_1:.1f})"))
            heatmap_row["RSI Overbought"] = f"{rsi_1:.2f}" # Store exact RSI value

        if "MACD Bearish Crossover" in selected_bearish and macd_bearish_crossover:
            signals.append((ticker, "bearish", f"📉 Bearish - MACD Bearish Crossover — {company}"))
            heatmap_row["MACD Bearish Crossover"] = "✔" # Checkmark for signal
        
        if "Death Cross" in selected_bearish and death_cross:
            signals.append((ticker, "bearish", f"💀 Bearish - Death Cross — {company}"))
            heatmap_row["Death Cross"] = "✔" # Checkmark for signal
        
        # --- New Bearish Confirmation Strategy ---
        if "Death Cross + RSI Bearish" in selected_bearish:
            if death_cross and (rsi_1 > 70):
                signals.append((ticker, "bearish", f"💀 Bearish - Death Cross + RSI Confirmed — {company}"))
                heatmap_row["Death Cross + RSI Bearish"] = "✔" # Checkmark for signal

        heatmap_data.append(heatmap_row)

    except Exception as e:
        st.error(f"❌ Error processing {ticker} ({company}): {e}")
        print(f"DEBUG: Unhandled error for {ticker}: {e}") # This will print to your terminal


# --- Add a Ticker Selector for Charting ---
st.markdown("---") # Separator for clarity
st.subheader("📊 Individual Stock Chart Analysis")
selected_chart_ticker = st.selectbox("Select Ticker for Detailed Chart", TICKERS, key="chart_ticker_select")

# Function to generate and display the chart
def plot_stock_chart(ticker_symbol, company_name):
    st.write(f"Displaying chart for **{ticker_symbol}** ({company_name})")

    # Fetch data for charting - longer period might be desired for visual trends
    # yfinance 5m interval is typically limited to 60 days
    chart_start_date = datetime.datetime.now() - datetime.timedelta(days=60) # Keep 60 days for 5m interval
    chart_end_date = datetime.datetime.now()

    try:
        chart_df = yf.download(ticker_symbol, start=chart_start_date, end=chart_end_date, interval="5m")

        if chart_df.empty or 'Close' not in chart_df.columns:
            st.warning(f"⚠️ No valid chart data available for {ticker_symbol} ({company_name}) for the selected period/interval.")
            return

        # Calculate indicators for charting (same as signal calculation, but for the chart_df)
        chart_df['20_MA'] = chart_df['Close'].rolling(window=20).mean()
        chart_df['50_MA'] = chart_df['Close'].rolling(window=50).mean()
        chart_df['200_MA'] = chart_df['Close'].rolling(window=200).mean()

        delta = chart_df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        chart_df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = chart_df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = chart_df['Close'].ewm(span=26, adjust=False).mean()
        chart_df['MACD'] = exp1 - exp2
        chart_df['MACD_Signal'] = chart_df['MACD'].ewm(span=9, adjust=False).mean()
        chart_df['MACD_Hist'] = chart_df['MACD'] - chart_df['MACD_Signal']


        # Create subplots: Price, MACD, RSI, Volume
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.15, 0.15, 0.2] # Proportions for each subplot height
        )

        # --- 1. Candlestick chart + Moving Averages ---
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='Candlestick',
            increasing_line_color='green', increasing_fillcolor='green',
            decreasing_line_color='red', decreasing_fillcolor='red'
        ), row=1, col=1)

        # Moving Averages - Check if the series contains any valid (non-NaN) data before plotting
        if not chart_df['20_MA'].empty and chart_df['20_MA'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['20_MA'], line=dict(color='orange', width=1), name='20 MA', legendgroup='MA'), row=1, col=1)
        if not chart_df['50_MA'].empty and chart_df['50_MA'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['50_MA'], line=dict(color='blue', width=1), name='50 MA', legendgroup='MA'), row=1, col=1)
        
        # --- REVISED 200 MA ADDITION LOGIC (Most likely fix for your issue) ---
        # Check if the 200_MA Series is not empty AND contains at least one non-NaN value.
        if not chart_df['200_MA'].empty and chart_df['200_MA'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['200_MA'], line=dict(color='purple', width=1), name='200 MA', legendgroup='MA'), row=1, col=1)


        # --- 2. MACD Subplot ---
        if not chart_df['MACD'].empty and chart_df['MACD'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD'], line=dict(color='green', width=1), name='MACD Line', legendgroup='MACD'), row=2, col=1)
        if not chart_df['MACD_Signal'].empty and chart_df['MACD_Signal'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD_Signal'], line=dict(color='red', width=1), name='Signal Line', legendgroup='MACD'), row=2, col=1)
        # MACD Histogram
        if not chart_df['MACD_Hist'].empty and chart_df['MACD_Hist'].dropna().any():
            macd_histogram_colors = ['rgba(0,128,0,0.7)' if val >= 0 else 'rgba(255,0,0,0.7)' for val in chart_df['MACD_Hist']]
            fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['MACD_Hist'], name='MACD Hist', marker_color=macd_histogram_colors, legendgroup='MACD'), row=2, col=1)


        # --- 3. RSI Subplot ---
        if not chart_df['RSI'].empty and chart_df['RSI'].dropna().any():
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], line=dict(color='darkorange', width=1.5), name='RSI', legendgroup='RSI'), row=3, col=1)
            # RSI Overbought/Oversold lines (always add if RSI itself is plotted)
            fig.add_trace(go.Scatter(x=chart_df.index, y=[70] * len(chart_df), line=dict(color='grey', width=1, dash='dash'), name='RSI Overbought', legendgroup='RSI'), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=[30] * len(chart_df), line=dict(color='grey', width=1, dash='dash'), name='RSI Oversold', legendgroup='RSI'), row=3, col=1)


        # --- 4. Volume Subplot ---
        if not chart_df['Volume'].empty and chart_df['Volume'].dropna().any():
            volume_colors = ['rgba(0,128,0,0.5)' if chart_df['Close'].iloc[i] > chart_df['Open'].iloc[i] else 'rgba(255,0,0,0.5)' for i in range(len(chart_df))]
            fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], name='Volume', marker_color=volume_colors, legendgroup='Volume'), row=4, col=1)


        # --- Update Layout and Axes ---
        fig.update_layout(
            title=f'{ticker_symbol} ({company_name}) Interactive Chart (5-Minute Interval)',
            xaxis_rangeslider_visible=False,
            height=900,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Update Y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)

        # Remove whitespace between subplots and adjust margins
        fig.update_layout(margin=dict(t=50, b=20, l=20, r=20),
                          xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                          yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))


        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generating chart for {ticker_symbol} ({company_name}): {e}")
        print(f"DEBUG: Chart generation error for {ticker_symbol}: {e}")

# Call the function for the selected ticker
if selected_chart_ticker:
    plot_stock_chart(selected_chart_ticker, TICKER_NAMES.get(selected_chart_ticker, selected_chart_ticker))


# --- Signal Display (Existing Logic) ---
if signals:
    st.markdown("### ✅ Current Trade Signals")
    for _, signal_type, msg in signals:
        if signal_type == "bullish":
            st.success(msg)
        elif signal_type == "bearish":
            st.error(msg)
else:
    st.info("No trade signals at this time for any active strategies.")

# --- Heatmap Matrix + Visual (Existing Logic) ---
if heatmap_data:
    st.markdown("### 🧭 Strategy Signal Matrix")

    heatmap_df = pd.DataFrame(heatmap_data)
    for strat in bullish_strategies + bearish_strategies:
        if strat not in heatmap_df.columns:
            heatmap_df[strat] = "" # Initialize with empty string

    def to_numeric_signal(val):
        return 1 if val != "" else 0

    numeric_heatmap_df = heatmap_df[bullish_strategies + bearish_strategies].applymap(to_numeric_signal)
    heatmap_df["Bullish Total"] = numeric_heatmap_df[bullish_strategies].sum(axis=1)
    heatmap_df["Bearish Total"] = numeric_heatmap_df[bearish_strategies].sum(axis=1)

    ordered_cols = ["Label"] + bullish_strategies + ["Bullish Total"] + bearish_strategies + ["Bearish Total"]
    heatmap_df = heatmap_df[ordered_cols]

    def highlight_bearish_total(val):
        if val >= 1:
            return 'background-color: salmon'
        return ''

    st.dataframe(
        heatmap_df.style
        .highlight_max(axis=0, subset=["Bullish Total"], color="lightgreen")
        .applymap(highlight_bearish_total, subset=['Bearish Total'])
    )

    st.markdown("### 🔥 Strategy Activation Heatmap")

    matrix = heatmap_df.set_index("Label")[bullish_strategies + bearish_strategies]

    def custom_color_from_string(val, strat):
        if val == "":
            return 0.5 
        
        if strat in bullish_strategies: 
            return 1.0 
        elif strat in bearish_strategies: 
            return 0.0 
        
        return 0.5 

    matrix_for_colors = matrix.copy()
    for col in matrix_for_colors.columns:
        matrix_for_colors[col] = matrix_for_colors[col].apply(lambda v: custom_color_from_string(v, col))

    custom_colorscale = [
        [0.0, "lightcoral"],
        [0.5, "#eeeeee"],
        [1.0, "lightgreen"]
    ]

    fig = px.imshow(
        matrix_for_colors,
        color_continuous_scale=custom_colorscale,
        text_auto=True,
        aspect="auto"
    )
    
    fig.update_traces(text=matrix.values, texttemplate="%{text}")

    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)
