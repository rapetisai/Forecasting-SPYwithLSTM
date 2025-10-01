# Forecasting SPY with LSTMs: Price Regression & Direction Classification

## 📌 Project Summary
This project builds and evaluates **Long Short-Term Memory (LSTM)** neural networks for forecasting the **SPY ETF** using historical OHLCV data and technical indicators (moving averages, RSI, MACD, Bollinger Bands).

Two complementary tasks are tackled:

1. **Regression — Predicting the next-day closing price**
   - Evaluated with RMSE, MAE, and MAPE.
   - Compared against naïve and linear baselines.

2. **Classification — Predicting the next-day market direction (up vs down)**
   - Evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC.
   - Compared against logistic regression and majority-class baselines.

The pipeline includes:
- Chronological data splitting (no leakage).
- Feature scaling with separate fit for train/test.
- 60-day rolling windows as input sequences.
- LSTM architectures tuned for each task.
- Visualizations of predicted vs actual prices and classification performance.

**Key Outcome**  
The models demonstrate that while LSTMs capture some temporal structure in SPY data, forecasting financial markets remains challenging—results highlight the importance of rigorous baselines, leakage prevention, and careful evaluation.

---

## 📊 Results (placeholders)
- **Regression**  
  RMSE = `XX.X` | MAE = `YY.Y` | MAPE = `ZZ%`  
  ![Actual vs Predicted](artifacts/actual_vs_predicted.png)

- **Classification**  
  Accuracy = `0.XX` | F1 = `0.XX` | ROC-AUC = `0.XX`  
  ![Confusion Matrix](artifacts/confusion_matrix.png)  
  ![ROC Curve](artifacts/roc_curve.png)

---

## ⚙️ Tech Stack
- Python, NumPy, Pandas
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Matplotlib, Plotly
- Alpha Vantage API 

---
