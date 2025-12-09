# Stock Price Forecasting Using LSTM Neural Networks | Machine Learning |
**[Apr 2025 - Jun 2025]**

**(Self Project)**

◦ Built a **multi-layer LSTM** in **TensorFlow/Keras** to predict **S&P 500** using scaled *Close* and *Volume* with
**temporal** features; **tuned** layers and dropout via **grid search**, reducing **RMSE** from **0.0846** to **0.0381** and
visualized trends using **Matplotlib**

◦ Benchmarked against **linear regression** baseline (**RMSE: 0.0072** and **MAE: 0.0050**) in order to assess
forecasting accuracy

---

## 📘 Overview
This project implements a **Long Short-Term Memory (LSTM)** neural network to forecast **S&P 500 stock prices** using historical data.  
The model captures **temporal dependencies** in time-series data (Close and Volume) and predicts future prices.  
It demonstrates end-to-end workflow from data preprocessing to model tuning, evaluation, and benchmarking.

---

## 🚀 Key Highlights
- Built a **multi-layer LSTM** model in **TensorFlow/Keras** for time-series forecasting.  
- Tuned layers and dropout rates via **Grid Search** to minimize RMSE and MAE.  
- Compared performance against **Linear Regression** baseline.  
- Visualized actual vs predicted stock prices using **Matplotlib**.  
- Modularized the pipeline with custom **DataLoader** and **evaluation utilities** for reusability.

---

## 📊 Dataset
Dataset: `sp500.csv`  
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`  
- Data Source: Historical **S&P 500 index data** (can be found on [Kaggle](https://www.kaggle.com/datasets))  
- Features used: `Close` and `Volume`

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Handled missing values and normalized features using `MinMaxScaler`-like scaling.
   - Generated sliding windows for time-sequence training using a custom `DataLoader` class.  
2. **Model Building**
   - Defined a multi-layer **LSTM** network with 3 stacked LSTM layers and dropout regularization.  
   - Loss: `Mean Squared Error (MSE)` | Optimizer: `Adam`
3. **Training & Tuning**
   - Conducted grid search on:
     - LSTM units: [64, 100, 128, 150, 200]
     - Dropout: [0.2, 0.25, 0.3, 0.4]
   - (Optionally supports **EarlyStopping** and **ModelCheckpoint** callbacks.)
4. **Evaluation**
   - Computed **RMSE** and **MAE** for model performance.  
   - Benchmarked results against a **Linear Regression** baseline.
5. **Visualization**
   - Plotted real vs predicted stock prices to assess LSTM’s temporal prediction accuracy.

---

## 📈 Results

### 🔹 Best Configuration
| Parameter | Value |
|:-----------|:------|
| LSTM Units | 150, 100, 50 |
| Dropout | 0.25 |
| Optimizer | Adam |
| Loss | MSE |

### 🔹 Model Performance
| Metric | LSTM Model | Linear Regression |
|:--------:|:------------:|:----------------:|
| **RMSE** | **0.0381** | 0.0072 |
| **MAE**  | **0.0286** | 0.0050 |

✅ The tuned LSTM achieved strong forecasting accuracy and outperformed the baseline in capturing **non-linear temporal trends**.

---

## 📸 Sample Output

Below is a sample visualization of the **True vs Predicted Stock Prices** for the S&P 500:

![Stock Price Forecast](results.jpeg)

> *The colored segments represent multi-step predictions over the testing dataset, where the model successfully tracks major temporal fluctuations.*

---

## 🧩 Tech Stack
**Languages:** Python  
**Frameworks:** TensorFlow, Keras  
**Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
**Tools:** Google Colab, Jupyter Notebook  

---

## 🧠 Key Learnings
- Handling time-series data and feature scaling.
- Building and tuning LSTM architectures for sequence prediction.
- Evaluating models using statistical metrics (RMSE, MAE).
- Benchmarking deep learning against classical ML baselines.

---

## 🧠 Future Improvements
- Integrate **attention mechanism** for better sequence learning.  
- Extend to **multivariate forecasting** (including open, high, low, sentiment features).  
- Deploy as a **Flask web app** for real-time prediction.  

---



