import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as pd_tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AAPL Stock Predictor", layout="wide")

st.title("🧠 AAPL Stock Price Prediction using Deep Learning")
st.write("Is web app mein pre-trained Deep Learning (LSTM) model ke zariye 7-day future stock forecast dikhaya gaya hai.")

# 1. Data Load
@st.cache_data
def load_data():
    data = pd.read_csv("AAPL_stock_data.csv")
    data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    df = data[data["Date"] >= "2015-01-01"][["Date", "adjclose"]].copy()
    return df

# 2. Model & Scaler Load
@st.cache_resource
def load_files():
    model = load_model("aapl_lstm_model.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    df = load_data()
    model, scaler = load_files()
    
    st.subheader("📊 Recent Historical Stock Data")
    st.dataframe(df.tail(5), use_container_width=True)

    time_step = 60
    
    # Preprocessing & Scaling
    scaled_data = scaler.transform(df[["adjclose"]])

    # Test Data Predictions
    X_test, y_test = [], []
    for i in range(time_step, len(scaled_data)):
        X_test.append(scaled_data[i - time_step:i, 0])
        y_test.append(scaled_data[i, 0])
        
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    actual_inv = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

    # Future 7-Day Forecasting
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_predictions = []
    curr_seq = last_60_days.copy()

    for _ in range(7):
        pred = model.predict(curr_seq, verbose=0)
        future_predictions.append(pred[0, 0])
        curr_seq = np.append(curr_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="B")

    # Visualizations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Stock Price Forecast Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        test_dates = df["Date"].iloc[time_step:]
        
        ax.plot(test_dates, actual_inv, label="Actual Price", color="blue")
        ax.plot(test_dates, predictions_inv, label="Model Predictions", color="orange", linestyle="--")
        ax.plot(future_dates, future_prices, label="7-Day Future Forecast", color="red", marker="o", linewidth=2)
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("🔮 Next 7 Days Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates.strftime('%Y-%m-%d'),
            "Predicted Price ($)": future_prices.flatten().round(2)
        })
        st.table(forecast_df)

except Exception as e:
    st.error(f"Error loading files: {e}")
