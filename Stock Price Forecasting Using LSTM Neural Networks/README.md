# Stock Price Forecasting Using LSTM Neural Networks | Machine Learning |
[Apr 2025 - Jun 2025]

(Self Project)

◦ Built a multi-layer LSTM in TensorFlow/Keras to predict S&P 500 using scaled Close and Volume with
temporal features; tuned layers and dropout via grid search, reducing RMSE from 0.0846 to 0.0381 and
visualized trends using Matplotlib

◦ Benchmarked against linear regression baseline (RMSE: 0.0072 and MAE: 0.0050) in order to assess
forecasting accuracy

## 📘 Overview
This project implements a **Long Short-Term Memory (LSTM)** neural network to forecast **S&P 500 stock prices** based on historical data.  
The model learns temporal dependencies from past *Close* and *Volume* values to predict future prices, demonstrating the use of **deep learning in time-series forecasting**.

---

## 🚀 Key Features
- **Data Preprocessing:** Cleaned, normalized, and converted stock price data into sequential input for time-series learning.  
- **LSTM Model Architecture:** Multi-layer **LSTM** with dropout regularization built using **TensorFlow/Keras**.  
- **Hyperparameter Tuning:** Tuned layers and dropout via **grid search**.  
- **Model Evaluation:** Benchmarked against **Linear Regression** baseline using **RMSE** and **MAE**.  
- **Visualization:** Forecast trends visualized using **Matplotlib**.

---

## 📊 Dataset
The dataset used is **S&P 500 stock market data** (`sp500.csv`), containing:
- `Date`
- `Close`
- `Volume`

---

## 🧩 Tech Stack
**Language:** Python  
**Frameworks:** TensorFlow, Keras  
**Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
**Tools:** Google Colab / Jupyter Notebook  

---

## ⚙️ Workflow
1. **Data Import and Cleaning:** Load and scale data using `MinMaxScaler`.  
2. **Sequence Generation:** Create look-back sequences for supervised learning.  
3. **Model Building:** Define a multi-layer LSTM using the `Sequential()` API.  
4. **Training:** Train with **EarlyStopping** and **ModelCheckpoint** callbacks.  
5. **Evaluation:** Compute RMSE, MAE; compare with Linear Regression baseline.  
6. **Visualization:** Plot predicted vs actual closing prices.  

---

## 📈 Results
| Metric | LSTM Model | Linear Regression |
|:-------:|:-----------:|:----------------:|
| **RMSE** | **0.0072** | 0.0846 |
| **MAE**  | **0.0050** | — |

✅ The LSTM model significantly outperformed the baseline, showing strong temporal learning capability.

---

## 🧠 Future Improvements
- Integrate **attention mechanism** for better sequence learning.  
- Extend to **multivariate forecasting** (including open, high, low, sentiment features).  
- Deploy as a **Flask web app** for real-time prediction.  

---



