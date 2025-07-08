import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go

# --- UI ---
st.set_page_config(layout="wide")
st.title("📈 Day Trading Dashboard")
strategy = st.sidebar.selectbox("Select Strategy", ["Breakout", "Scalping", "Trend Trading"])
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()

# --- Time Settings ---
now = datetime.datetime.now()
start_date = now - datetime.timedelta(days=5)
end_date = now

# --- Load Data ---
st.subheader(f"Loading data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date, interval="5m")
data.dropna(inplace=True)

# --- Indicators ---
data['20_MA'] = data['Close'].rolling(window=20).mean()
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['High_Break'] = data['High'].rolling(window=20).max()
data['Low_Break'] = data['Low'].rolling(window=20).min()
data['Volume_Surge'] = data['Volume'] > data['Volume'].rolling(window=20).mean() * 1.5

# --- Signal Logic ---
signal = ""
if strategy == "Breakout":
    recent_high = data['High_Break'].iloc[-1]
    if data['Close'].iloc[-1] > recent_high:
        signal = f"🔔 Breakout Alert: {ticker} is breaking above recent resistance at ${recent_high:.2f}"
elif strategy == "Scalping":
    if data['20_MA'].iloc[-1] > data['50_MA'].iloc[-1] and data['Volume_Surge'].iloc[-1]:
        signal = f"⚡ Scalping Opportunity: {ticker} has volume surge + short-term MA crossover"
elif strategy == "Trend Trading":
    if data['20_MA'].iloc[-1] > data['50_MA'].iloc[-1]:
        signal = f"📈 Trend Trade: {ticker} in uptrend (20 MA > 50 MA). Consider riding the wave."

# --- Display Signal ---
if signal:
    st.success(signal)
else:
    st.info("No clear signal based on current strategy.")

# --- Chart ---
st.subheader("📊 Price Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlestick'))
fig.add_trace(go.Scatter(x=data.index, y=data['20_MA'], mode='lines', name='20 MA'))
fig.add_trace(go.Scatter(x=data.index, y=data['50_MA'], mode='lines', name='50 MA'))
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Display Table ---
st.subheader("📋 Latest Data")
st.dataframe(data.tail(15))
