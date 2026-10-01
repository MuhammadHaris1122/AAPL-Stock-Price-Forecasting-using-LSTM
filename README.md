# 📈 AAPL Stock Price Forecasting using Deep Learning (LSTM)

An end-to-end Deep Learning time-series forecasting project that predicts Apple Inc. (AAPL) stock prices and provides a 7-day future price trend forecast using stacked Long Short-Term Memory (LSTM) networks.

## 🚀 Live Demo
[Click here to view the Streamlit Web App](APNI_STREAMLIT_APP_KA_LINK_HERE)

## 📌 Features
- **Deep Learning Model**: Built with TensorFlow/Keras using stacked LSTM layers and Dropout regularization.
- **Historical Data Analysis**: Preprocessed stock data (2015–Present) utilizing Adjusted Close prices to eliminate stock split distortions.
- **7-Day Future Forecast**: Recursive single-step forecasting loop to predict prices for the upcoming 7 business trading days.
- **Interactive UI**: Clean Streamlit dashboard displaying historical predictions alongside actual stock prices and future forecasts.

## 🛠️ Tech Stack
- **Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Web Framework**: Streamlit
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib, Scikit-Learn
- **Model Serialization**: Joblib

## 📊 Dataset
The dataset contains historical stock market data for Apple Inc. (`AAPL`) including Open, High, Low, Close, Adjusted Close, and Volume.
