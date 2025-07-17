# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Crypto Forecaster",
    page_icon="🔮",
    layout="wide"
)


# --- Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    """Loads and preprocesses data for all cryptocurrencies."""
    crypto_files = {'BTC': 'coin_Bitcoin.csv', 'ETH': 'coin_Ethereum.csv', 'LTC': 'coin_Litecoin.csv'}
    crypto_data = {}
    for symbol, file_path in crypto_files.items():
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            crypto_data[symbol] = df
        except FileNotFoundError:
            st.error(f"Error: The file {file_path} was not found.")
            st.stop()
    return crypto_data


def standardize_data(df):
    df = df[['Date', 'Close']].copy()
    df.sort_values(by='Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'].dt.date)
    df.set_index('Date', inplace=True)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)
    df.ffill(inplace=True)
    return df


def create_advanced_features(df):
    df_feat = df.copy()
    df_feat['Target'] = (df_feat['Close'].shift(-1) / df_feat['Close']) - 1
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = ema_12 - ema_26
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    df_feat['bollinger_mid'] = df_feat['Close'].rolling(window=20).mean()
    rolling_std = df_feat['Close'].rolling(window=20).std()
    df_feat['bollinger_upper'] = df_feat['bollinger_mid'] + (rolling_std * 2)
    df_feat['bollinger_lower'] = df_feat['bollinger_mid'] - (rolling_std * 2)
    df_feat['lag_1'] = df_feat['Close'].shift(1)
    df_feat['rolling_mean_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat.dropna(inplace=True)
    return df_feat


def create_sequences(X_data, time_steps=60):
    Xs = []
    for i in range(len(X_data) - time_steps + 1):
        Xs.append(X_data[i:(i + time_steps)])
    return np.array(Xs)


# --- Model Training and Backtesting (Cached) ---
@st.cache_resource
def run_pipeline(symbol, X, y, df_close):
    """Performs the full data split, training, and returns all necessary artifacts."""
    st.write(f"Cache miss: Running full pipeline for {symbol}...")
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, objective='reg:squarederror', n_jobs=-1,
                                 random_state=42, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # LSTM
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    TIME_STEPS = 60
    X_train_seq = create_sequences(X_train_scaled, TIME_STEPS)
    y_train_seq = y_train_scaled[TIME_STEPS - 1:]

    lstm_model = Sequential(
        [tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])), LSTM(50, return_sequences=False),
         Dropout(0.2), Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32, verbose=0)

    # Return everything needed for the backtest
    return {
        'xgb': xgb_model, 'lstm': lstm_model,
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'data': {
            'X_test': X_test,
            'df_close': df_close,
            'split_point': split_point,
            'TIME_STEPS': TIME_STEPS
        }
    }


@st.cache_data
def run_backtest(symbol, _pipeline_results):
    """Runs the backtest using the results from the pipeline."""
    st.write(f"Cache miss: Running backtest for {symbol}...")
    # Unpack all artifacts
    xgb_model, lstm_model = _pipeline_results['xgb'], _pipeline_results['lstm']
    scaler_X, scaler_y = _pipeline_results['scalers']['X'], _pipeline_results['scalers']['y']
    X_test, df_close, split_point, TIME_STEPS = _pipeline_results['data']['X_test'], _pipeline_results['data'][
        'df_close'], _pipeline_results['data']['split_point'], _pipeline_results['data']['TIME_STEPS']

    # Generate predictions
    xgb_predicted_returns = xgb_model.predict(X_test)
    X_test_scaled = scaler_X.transform(X_test)
    X_test_seq = create_sequences(X_test_scaled, TIME_STEPS)
    lstm_predicted_returns_scaled = lstm_model.predict(X_test_seq, verbose=0)
    lstm_predicted_returns = scaler_y.inverse_transform(lstm_predicted_returns_scaled).flatten()

    # Align predictions and actual prices
    actual_prices = df_close[split_point + TIME_STEPS - 1:]
    xgb_returns_aligned = xgb_predicted_returns[TIME_STEPS - 1:]

    if len(xgb_returns_aligned) != len(lstm_predicted_returns) or len(actual_prices) != len(lstm_predicted_returns):
        st.error("Fatal alignment error during backtest. Please clear cache and rerun.")
        st.stop()

    # Create and run backtest
    ensemble_returns = (xgb_returns_aligned + lstm_predicted_returns) / 2
    backtest_df = pd.DataFrame({'Actual_Price': actual_prices, 'Predicted_Return': ensemble_returns})
    backtest_df['Actual_Return'] = backtest_df['Actual_Price'].pct_change()
    backtest_df['Signal'] = np.where(backtest_df['Predicted_Return'] > 0, 1, 0)
    backtest_df['Strategy_Return'] = backtest_df['Signal'].shift(1) * backtest_df['Actual_Return']
    backtest_df.dropna(inplace=True)

    backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Return']).cumprod()
    backtest_df['Cumulative_Buy_Hold_Return'] = (1 + backtest_df['Actual_Return']).cumprod()

    # Calculate final metrics
    metrics = {
        'strategy_return': (backtest_df['Cumulative_Strategy_Return'].iloc[
                                -1] - 1) * 100 if not backtest_df.empty else 0,
        'buy_hold_return': (backtest_df['Cumulative_Buy_Hold_Return'].iloc[
                                -1] - 1) * 100 if not backtest_df.empty else 0,
        'win_rate': (backtest_df['Strategy_Return'] > 0).mean() * 100 if not backtest_df.empty else 0
    }
    return metrics, backtest_df


# --- Streamlit UI ---
st.title("Advanced Cryptocurrency Forecaster 🔮")
st.write(
    "This dashboard analyzes historical data, backtests an advanced ensemble model (XGBoost + LSTM), and presents the potential investment insights.")

raw_data = load_data()

selected_symbol = st.selectbox("Choose a Cryptocurrency to Analyze:", options=['BTC', 'ETH', 'LTC'],
                               format_func=lambda x: {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'LTC': 'Litecoin'}[x])

if selected_symbol:
    with st.spinner(f"Running full pipeline for {selected_symbol}... This may take several minutes the first time."):
        df_std = standardize_data(raw_data[selected_symbol])
        df_advanced = create_advanced_features(df_std)
        features = ['macd', 'macd_signal', 'bollinger_mid', 'bollinger_upper', 'bollinger_lower', 'lag_1',
                    'rolling_mean_30']
        X = df_advanced[features]
        y = df_advanced['Target']

        # Run pipeline and get all artifacts
        pipeline_results = run_pipeline(selected_symbol, X, y, df_advanced['Close'])
        metrics, backtest_df = run_backtest(selected_symbol, pipeline_results)

    # --- Display Results ---
    st.header(f"Backtesting Results for {selected_symbol}")
    st.write(
        f"This simulation tests a strategy from **{backtest_df.index.min().strftime('%Y-%m-%d')}** to **{backtest_df.index.max().strftime('%Y-%m-%d')}**.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Strategy Return", f"{metrics['strategy_return']:.2f}%")
    col2.metric("Buy & Hold Return", f"{metrics['buy_hold_return']:.2f}%")
    col3.metric("Strategy Win Rate", f"{metrics['win_rate']:.2f}%")

    st.subheader("Strategy Performance vs. Buy and Hold")
    fig_equity = px.line(backtest_df, y=['Cumulative_Strategy_Return', 'Cumulative_Buy_Hold_Return'],
                         labels={'value': 'Growth of $1', 'variable': 'Strategy'},
                         color_discrete_map={'Cumulative_Strategy_Return': '#1f77b4',
                                             'Cumulative_Buy_Hold_Return': '#ff7f0e'})
    st.plotly_chart(fig_equity, use_container_width=True)

    st.info(
        "Disclaimer: This is a financial data science project, not investment advice. Past performance is not indicative of future results.",
        icon="⚠️")