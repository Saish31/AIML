# Intraday Mean-Reversion Strategy with Rolling GARCH Volatility Forecasts


## Project Overview
This repository contains an end-to-end backtest of an intraday mean-reversion strategy that:

1. **Forecasts daily volatility regimes** using a rolling-window GARCH(1,3) model.  
2. **Generates intraday entry signals** via RSI and Bollinger Band extremes on 5-minute bars.  
3. **Executes trades** only when daily and intraday signals align in a mean-reversion context.  
4. **Backtests performance** on historical data and produces an equity curve and risk metrics.

---

## Key Features
- **Time-series volatility modeling** (GARCH forecasts)  
- **Technical indicator signals** (RSI, Bollinger Bands)  
- **Regime-filtered entries** to improve signal quality  
- **Vectorized pandas implementation** for efficiency  
- **Daily and intraday aggregation** with clear P&L attribution  
- **Extensible framework** for transaction cost, parameter sweeps, and walk-forward tests  

