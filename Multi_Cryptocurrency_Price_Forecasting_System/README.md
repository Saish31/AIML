# Advanced Cryptocurrency Forecasting & Strategy Backtesting Engine

---

### ## 🔮 Live Dashboard

### ## 📸 Dashboard Screenshot
<img width="263" height="278" alt="image" src="https://github.com/user-attachments/assets/112cfe50-e3c7-4e9a-aa2e-b7a87d2d1711" />

![Dashboard Screenshot](path/to/your/screenshot.png)
*(Replace the above path with a screenshot of your running application)*

---

### ## ✨ Key Features

* **Multi-Asset Analysis:** Supports analysis for Bitcoin (BTC), Ethereum (ETH), and Litecoin (LTC).
* **Comparative Modeling:** Rigorously builds, trains, and evaluates multiple model classes: Statistical (SARIMAX), Gradient Boosting (XGBoost), and Deep Learning (LSTM).
* **Advanced Feature Engineering:** Creates a rich feature set including MACD, Bollinger Bands, and various lags and rolling statistics.
* **Ensemble Forecasting:** Combines predictions from the best-performing models to create a more robust final forecast.
* **Model Interpretability:** Uses **SHAP (SHapley Additive exPlanations)** to understand the "why" behind the model's predictions, revealing learned financial patterns like momentum and mean-reversion.
* **Quantitative Backtesting:** Translates model predictions into financial outcomes by simulating a trading strategy and comparing its performance against a "Buy and Hold" benchmark.
* **Interactive Dashboard:** A user-friendly web application built with Streamlit, featuring performance caching for a fast user experience.

---

### ## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Deep Learning:** TensorFlow / Keras
* **Time Series:** pmdarima (for SARIMAX baseline)
* **Interpretability:** SHAP
* **Dashboard & Visualization:** Streamlit, Plotly

---

### ## 🔬 The Data Science Pipeline

This project follows a rigorous, iterative data science workflow.

1.  **Data Preprocessing:** Raw time-series data for each cryptocurrency is loaded, cleaned, and standardized to ensure a continuous daily timeline without missing values.

2.  **Feature Engineering & Target Transformation:** The project's core insight was that predicting raw, non-stationary prices is ineffective for ML models. The pipeline was re-engineered to predict **stationary daily returns**. A sophisticated feature set was built, including standard lags and advanced technical indicators like MACD and Bollinger Bands.

3.  **Modeling & Evaluation:** Multiple models were trained on this rich dataset. An initial comparison revealed that the advanced models struggled, leading to the pivot to modeling returns. After the pivot, a new leaderboard was established:
    | Model | MAPE (on Price) |
    | :--- | :--- |
    | **Ensemble (XGBoost + LSTM)** | **3.11%** |
    | XGBoost | 3.41% |
    | LSTM | 3.78% |
    | SARIMAX (Baseline) | 4.15% |

4.  **Strategy Backtesting & Conclusion:** The final ensemble model's predictions were used to power a simple trading strategy. The backtest revealed that while the model had a **win rate > 50%**, it underperformed a simple "Buy and Hold" strategy during the 2020-2021 bull market. This highlights a key challenge in financial ML: models trained on historical data may not generalize during new **market regime shifts**. The project's success lies in its ability to rigorously arrive at this nuanced and realistic conclusion.

---

### ## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
