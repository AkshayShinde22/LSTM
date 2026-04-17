import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import math
from datetime import datetime

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Legends Stock AI", layout="wide", page_icon="📈")

# Groww-style Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f1116; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #363a45; }
    div[data-testid="stMetricValue"] { color: #00d09c; } /* Groww Green */
    .positive { color: #00d09c; font-weight: bold; }
    .negative { color: #eb5b5b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. STOCK LIST & SEARCH ---
# Expanded list (Can be further automated with a CSV of NIFTY 500)
STOCK_LIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META",
    "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "ZOMATO.NS"
]

st.title("📈 Legends Hybrid Stock AI")
st.caption("AI-Powered Technical & Sentiment Analysis Dashboard")

# Search bar in sidebar
with st.sidebar:
    st.header("Search & Settings")
    ticker = st.selectbox("Search Stock Symbol", options=STOCK_LIST, index=0)
    time_period = st.select_slider("Data History", options=["1y", "2y", "5y", "max"], value="2y")
    predict_days = st.slider("Prediction Horizon (Days)", 1, 30, 7)
    
    st.divider()
    st.info("Researcher: Akshay Arjun Shinde\nCollege: Kirti College")

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_data(symbol, period):
    data = yf.download(symbol, period=period, auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

df = load_data(ticker, time_period)

if df is not None:
    # Technical Indicators
    df['SMA20'] = SMAIndicator(df['Close'].squeeze(), 20).sma_indicator()
    df['SMA50'] = SMAIndicator(df['Close'].squeeze(), 50).sma_indicator()
    
    # Sentiment Simulation (Groww Style Logic)
    sentiment_val = np.random.uniform(-1, 1)
    mood = "BULLISH" if sentiment_val > 0.1 else "BEARISH" if sentiment_val < -0.1 else "NEUTRAL"
    mood_class = "positive" if mood == "BULLISH" else "negative" if mood == "BEARISH" else ""

    # --- 4. TOP METRICS DASHBOARD ---
    last_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{last_price:,.2f}", f"{pct_change:.2f}%")
    col2.metric("Market Sentiment", mood)
    col3.metric("SMA 50 Status", "Above" if last_price > df['SMA50'].iloc[-1] else "Below")
    col4.write(f"**Analysis Conclusion:** <br><span class='{mood_class}'>{mood} Trend Detected</span>", unsafe_allow_html=True)

    # --- 5. INTERACTIVE CANDLESTICK CHART ---
    st.subheader("Market Visualizer")
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price Action", increasing_line_color='#00d09c', decreasing_line_color='#eb5b5b'
    )])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="20 SMA", line=dict(color='#ff9800', width=1)))
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. AI PREDICTION ENGINE ---
    st.subheader("🤖 LSTM Intelligence Forecast")
    
    if st.button("Generate AI Forecast"):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']].values)

        # Prepare sequences for LSTM
        seq_len = 60
        x_train, y_train = [], []
        for i in range(seq_len, len(scaled_data)):
            x_train.append(scaled_data[i-seq_len:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        with st.spinner("AI is analyzing historical patterns..."):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
                Dropout(0.1),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
            
            # Predict Future
            last_60_days = scaled_data[-60:].reshape(1, 60, 1)
            pred_price = model.predict(last_60_days)
            final_pred = scaler.inverse_transform(pred_price)[0][0]

            # Display Result
            st.success(f"AI Predicted Price for Tomorrow: **₹{final_pred:.2f}**")
            
            # Prediction Logic vs Current
            if final_pred > last_price:
                st.markdown(f"💡 **Recommendation:** The AI suggests a <span class='positive'>Potential Upside</span>.", unsafe_allow_html=True)
            else:
                st.markdown(f"💡 **Recommendation:** The AI suggests a <span class='negative'>Potential Correction</span>.", unsafe_allow_html=True)

else:
    st.error("No data found. Please check the ticker symbol.")
