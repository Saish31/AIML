# EUR/USD Forecasting with Transformer-Based Time Series Model

A deep learning project that leverages the Transformer architecture to predict hourly EUR/USD close prices from historical candlestick and technical indicator data.

## 📈 Overview

This project explores the use of Transformer-based sequence models for forecasting foreign exchange (FX) prices, specifically the EUR/USD currency pair on an hourly timeframe. The model takes sequences of engineered features — including OHLCV values and popular technical indicators — and forecasts the next hour's close price. It aims to assess whether attention-based architectures, which have revolutionized NLP, can capture useful patterns in noisy and non-stationary financial time series.

---

## 🔧 Features

- ✅ End-to-end pipeline: from raw CSV to forecasted price and backtested performance  
- ✅ Transformer model implemented from scratch using PyTorch  
- ✅ Technical indicator feature engineering with the `ta` library  
- ✅ Custom data preprocessing, normalization, and train/test splitting  
- ✅ Visual evaluation of predictions and trend alignment  
- ✅ Modular, reproducible, and ready for extension

---

---

## 🧠 Model Architecture

The forecasting model is based on a Transformer encoder architecture, adapted from NLP to financial time series:

- Positional encoding to represent hourly sequence order  
- Multi-head self-attention to capture short and long-range dependencies  
- Feed-forward layers with residual connections and layer normalization  
- Final dense layer to regress the next closing price

---

## 📊 Data

- **Source**: Hourly EUR/USD candlestick data  
- **Fields**: Open, High, Low, Close, Volume, Timestamp  
- **Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - Additional momentum, trend, and volatility features via `ta` library

---

## 📈 Evaluation Metrics

- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Directional Accuracy (up/down prediction)  
- Visualization: Real vs. predicted close prices  
- (Optional) Financial backtest with:
  - Cumulative returns
  - Sharpe ratio
  - Maximum drawdown

---

## 🧪 Key Results

- The Transformer captures general price direction but underestimates volatility.  
- Slight lag observed at market turning points.  
- Consistent downward prediction bias, suggesting room for calibration.  
- Predicted prices broadly align with actual trends, though peaks/troughs are dampened.



