# 📈 AAPL Stock Price Forecasting — Deep Learning vs. Baseline Comparison

## Overview
Built an end-to-end Deep Learning application using stacked LSTM networks to forecast Apple Inc. (`AAPL`) stock prices and predict a 7-day future price trend. The project evaluates time-series forecasting performance against traditional benchmarks to understand the limits of price-only historical data.

## Live Demo
🔗 [Try the Live Streamlit App]https://aapl-stock-price-forecasting-using-lstm-n9a3aswwginykla7pwcrex.streamlit.app/

## Dataset
- **Source**: Yahoo Finance (`AAPL` Adjusted Close Prices)
- **Date Range**: January 2015 – Present
- **Features**: Historical daily `adjclose` prices to account for stock splits and dividends.

## Approach
- **Data Preprocessing**: Feature scaling using `MinMaxScaler(0, 1)` fitted only on training data to prevent data leakage.
- **Window Size**: 60-day sliding sequence window ($t-60$ to $t$) to predict the next day ($t+1$).
- **Architecture**: Stacked LSTM layers with `Dropout (0.2)` regularization to prevent overfitting, trained using the `Adam` optimizer.
- **Train/Test Split**: Sequential 80% train / 20% test split to respect temporal order.
- **Multi-step Forecasting**: 7-day recursive prediction loop using autoregressive feedback.

## Results

Actual vs. Predicted Prices along with the 7-Day Recursive Forecast.*

| Metric | LSTM Model | Naive Baseline |
|--------|------------|----------------|
| **MAE** | $4.49 | **$1.96** |
| **R² Score** | 0.82 | — |

## Challenges & Key Learnings

### 1. Model vs. Naive Baseline
After training the LSTM model ($R^2 = 0.82$, $\text{MAE} = \$4.49$), I compared it against a naive baseline (predicting tomorrow's price = today's price), which achieved a lower MAE of $\$1.96$.

**Key Insight:** This result aligns with the Efficient Market Hypothesis — short-term stock prices are largely unpredictable from historical price data alone, since markets quickly price in available information. The LSTM model essentially learned a similar "persistence" pattern to the naive baseline, confirming that daily closing price history alone is insufficient to meaningfully outperform a simple baseline for short-term forecasting.

### 2. Autoregressive Error Accumulation in Multi-Day Forecasting
When performing the 7-day future recursive forecast, the model uses its own predicted values as inputs for subsequent steps. Since each prediction contains a small margin of error, feeding predicted values recursively leads to error compounding over time, causing the 7-day trend to flatten or drift over longer time horizons.

### 3. Data Leakage & Normalization Pitfalls
Initial experiments showed artificially high test accuracy when scaling the entire dataset before splitting. Fitting the `MinMaxScaler` exclusively on the training set and transforming the test set separately was critical to ensuring realistic, leak-free evaluations.

### Takeaway
This project highlights the importance of baseline comparison in time series forecasting — a high $R^2$ score alone can be misleading without contextualizing it against a simple benchmark. Future improvements would require incorporating external signals like market sentiment, trading volume, or macroeconomic indicators.

## Tech Stack
`Python` • `TensorFlow / Keras` • `Streamlit` • `Pandas` • `NumPy` • `Scikit-Learn` • `Matplotlib` • `Joblib`

## Disclaimer
This project is for educational and research purposes only and does not constitute financial advice.
