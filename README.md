# 📉 Topological Data Analysis for Market Crash Prediction

A machine learning–based **early-warning framework for financial market crash risk prediction** using **Topological Data Analysis (TDA), Persistent Homology, CatBoost, and SHAP Explainability**.

The project investigates whether changes in the **topological structure of multidimensional financial markets** can provide useful warning signals before significant market declines.

## 🌐 Live Application

The project is deployed as an interactive Streamlit dashboard.

**Live Demo:**  
https://tda-for-market-crash-prediction.streamlit.app/

The dashboard provides:

- Market risk overview
- Historical crash-risk predictions
- Interactive market analysis
- TDA feature analysis
- SHAP model explainability
- Historical backtesting
- Model performance evaluation
- Ablation study and methodology

---

## 📌 Project Overview

Financial market crashes are rare and complex events influenced by volatility, investor uncertainty, asset correlations, and structural changes across financial markets.

Traditional crash-prediction approaches generally rely on indicators such as:

- Returns
- Volatility
- Momentum
- Moving averages
- Market fear indicators such as VIX

While these features capture individual market behavior, they may not fully represent the **underlying structural shape of the market**.

This project introduces **Topological Data Analysis (TDA)** to capture these hidden structural patterns and combines them with conventional financial indicators for crash-risk prediction.

---

## 🎯 Problem Statement

The objective of this project is to develop an early-warning system capable of identifying market conditions associated with severe future declines.

The main research question is:

> **Can topological changes in multidimensional financial market behavior improve crash-risk prediction beyond traditional market indicators?**

To investigate this, the project compares conventional market features against features generated using **Persistent Homology**.

---

## 📊 Market Data

The system analyzes multiple financial markets:

- **S&P 500** – Broad U.S. equity market
- **NASDAQ** – Technology-focused equity market
- **VIX** – Market volatility/fear indicator
- **Gold** – Safe-haven asset
- **Crude Oil** – Commodity and macroeconomic indicator

Using multiple markets allows the model to analyze broader financial-system behavior rather than relying only on a single stock index.

---

## ⚙️ Project Workflow

```text
Financial Market Data
        ↓
Data Cleaning & Preprocessing
        ↓
Financial Feature Engineering
        ↓
Future 10-Day Drawdown Calculation
        ↓
Crash-Risk Label Generation
        ↓
30-Day Rolling Market Windows
        ↓
Persistent Homology
        ↓
TDA Feature Extraction
        ↓
Market Features + TDA Features
        ↓
CatBoost Classification
        ↓
Crash Probability
        ↓
Risk Threshold Classification
        ↓
SHAP Explainability
        ↓
Historical Backtesting
        ↓
Interactive Streamlit Dashboard
```

---

## 🧮 Financial Feature Engineering

The model uses conventional financial indicators including:

- Daily returns
- Log returns
- Rolling volatility
- Momentum indicators
- Moving-average distance
- VIX changes
- VIX moving averages
- VIX ratio
- Cross-market information

These features describe traditional market behavior and form the baseline model.

---

## 🔬 Topological Data Analysis

The key innovation of this project is the use of **Topological Data Analysis**.

Instead of analyzing only individual financial indicators, TDA studies the **shape and structure of multidimensional market behavior**.

Historical observations are transformed into rolling **30-day multidimensional market windows**.

Persistent Homology is then applied to extract structural information.

### H0 — Connected Components

H0 represents connectivity and clustering structure within the market point cloud.

### H1 — Loops

H1 captures cyclic or loop-like structures that may appear in multidimensional market relationships.

### Extracted TDA Features

Nine topological features are generated:

```text
TDA_H0_Count
TDA_H0_TotalPersistence
TDA_H0_MaxPersistence
TDA_H0_MeanPersistence

TDA_H1_Count
TDA_H1_TotalPersistence
TDA_H1_MaxPersistence
TDA_H1_MeanPersistence
TDA_H1_Entropy
```

These features are combined with traditional market indicators for machine learning.

---

## 🤖 Machine Learning Model

The final selected model uses:

**Market Features + TDA Features + CatBoost**

Total final features:

```text
29 Features
```

CatBoost was selected for its ability to model complex nonlinear relationships and interactions among financial and topological features.

### Experimental Components

The project also evaluated:

- **PELT Change-Point Detection**
- **Hidden Markov Model (HMM)** for market regimes

These methods were analyzed experimentally but were **not retained in the final selected architecture**, because the Market + TDA model provided stronger balanced rare-event prediction performance.

---

## 📈 Model Performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8365** |
| PR-AUC | **0.1547** |
| Precision @ 0.62 | **0.2065** |
| Recall @ 0.62 | **0.4634** |
| F1 Score @ 0.62 | **0.2857** |
| TDA SHAP Contribution | **38.12%** |

Because market crashes are rare events, **PR-AUC is particularly important** for evaluating the model.

The test crash prevalence/random PR baseline was approximately:

```text
0.0346
```

The final model achieved:

```text
PR-AUC = 0.1547
```

---

## 🧪 Ablation Study

Different feature combinations were evaluated to determine whether TDA genuinely improved prediction.

| Model | Features | ROC-AUC | PR-AUC | F1 @ 0.50 |
|---|---:|---:|---:|---:|
| Market Only | 20 | 0.7829 | 0.0965 | 0.1783 |
| **Market + TDA** | **29** | **0.8365** | **0.1547** | **0.2703** |
| Market + TDA + PELT | 35 | 0.8205 | 0.1127 | 0.2128 |
| Full Experimental Model | 36 | 0.8468 | 0.1301 | 0.1957 |

### Key Finding

Adding TDA improved:

```text
ROC-AUC
0.7829 → 0.8365

PR-AUC
0.0965 → 0.1547

F1
0.1783 → 0.2703
```

This provides evidence that **topological features contain useful predictive information beyond conventional market indicators**.

---

## 🧠 SHAP Explainability

SHAP was used to understand how individual features influence model predictions.

The most influential features include:

1. **TDA_H0_TotalPersistence**
2. SP500_Volatility_10
3. **TDA_H0_MeanPersistence**
4. VIX_Ratio
5. VIX
6. SP500_Volatility_20
7. NASDAQ_Volatility_20
8. VIX_MA10
9. **TDA_H0_MaxPersistence**
10. SP500_MA50_Distance

### Major Finding

```text
Overall TDA SHAP Contribution = 38.12%
```

**TDA_H0_TotalPersistence** emerged as the strongest global SHAP feature.

This indicates that topological changes in market connectivity contributed substantially to the model's predictions.

---

## 🚨 Crash Risk Classification

The predicted probability is converted into three risk levels.

| Probability | Risk Level |
|---|---|
| `< 0.37` | LOW |
| `0.37 – 0.62` | WARNING |
| `≥ 0.62` | HIGH RISK |

### Warning Threshold — 0.37

Designed for higher recall:

```text
Precision = 14.17%
Recall    = 82.93%
F1        = 24.20%
```

This threshold is useful as an early-warning signal where detecting more potential crash periods is prioritized.

### High-Risk Threshold — 0.62

Best F1 threshold:

```text
Precision = 20.65%
Recall    = 46.34%
F1        = 28.57%
```

---

## ⏳ Historical Backtesting

The model was evaluated on an out-of-sample period covering:

```text
October 2021 – July 2026
```

The backtesting framework analyzes:

- Crash-risk probability
- Warning signals
- High-risk signals
- Actual crash-risk periods
- Worst future drawdowns
- Warning lead time
- High-risk lead time
- Missed crash episodes

This allows the system to be evaluated as an **early-warning framework**, rather than only using standard classification metrics.

---

## 🖥️ Streamlit Dashboard

The interactive dashboard contains the following sections:

### Overview
Displays the latest available out-of-sample risk signal, model metrics, crash probability timeline, and risk distribution.

### Market Analysis
Interactive visualization of S&P 500, VIX, NASDAQ, Gold, Oil, volatility, momentum, and cross-market behavior.

### TDA Intelligence
Explains Persistent Homology and visualizes the contribution of topological features.

### Crash Prediction
Allows historical prediction dates to be inspected using LOW, WARNING, and HIGH risk classifications.

### Explainability
Displays global and local SHAP analysis to explain model predictions.

### Backtesting
Evaluates historical crash episodes, warning signals, and lead times.

### Model Performance
Provides ROC, Precision-Recall, confusion matrix, threshold analysis, and ablation results.

### Methodology
Explains the complete project pipeline and experimental methodology.

---

## 🛠️ Technologies Used

### Programming

- Python
- Pandas
- NumPy

### Machine Learning

- CatBoost
- Scikit-learn

### Topological Data Analysis

- Persistent Homology
- Vietoris-Rips Complex

### Experimental Analysis

- PELT Change-Point Detection
- Hidden Markov Models

### Explainable AI

- SHAP

### Visualization & Deployment

- Streamlit
- Plotly
- Matplotlib
- Streamlit Community Cloud

---

## 📂 Project Structure

```text
TDA-Market-Crash-Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── market_data_final_features.csv
│   ├── final_test_predictions.csv
│   ├── final_metrics.csv
│   ├── ablation_results.csv
│   ├── historical_backtest_daily.csv
│   ├── crash_episode_backtest.csv
│   └── shap_feature_importance.csv
│
├── outputs/
│   ├── evaluation/
│   └── shap/
│
└── README.md
```

The exact structure may contain additional preprocessing, feature-engineering, TDA, and model-training scripts.

---

## 🚀 Run Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live Application:**

https://tda-for-market-crash-prediction.streamlit.app/

---

## 🔍 Key Findings

- TDA significantly improved performance compared with traditional market features alone.
- PR-AUC increased from **0.0965 to 0.1547** after adding TDA.
- TDA features contributed approximately **38.12%** of global SHAP importance.
- **TDA_H0_TotalPersistence** was the strongest global predictive feature.
- Persistent Homology provides complementary structural information that conventional financial indicators may not capture.
- The Market + TDA architecture provided the strongest balanced performance among the evaluated configurations.

---

## ⚠️ Limitations

Market crashes are extremely rare and difficult to predict.

Therefore:

- False-positive warnings can occur.
- Model predictions represent probabilities, not certainties.
- Historical performance does not guarantee future performance.
- Financial market relationships can change over time.
- The model requires periodic retraining and validation.
- Saved dashboard predictions should not be interpreted as live market forecasts unless a real-time inference pipeline is implemented.

---

## 📌 Conclusion

This project demonstrates how **Topological Data Analysis can complement conventional financial machine learning** for market crash-risk prediction.

By combining traditional market indicators with structural features extracted through **Persistent Homology**, the final CatBoost model achieved stronger rare-event prediction performance than a market-only baseline.

The results suggest that changes in the **topological structure of financial markets** may contain useful information for identifying periods of elevated systemic instability.

---

## ⚠️ Disclaimer

This project is developed for **research and educational purposes only**.

It does not provide financial, investment, or trading advice. Model predictions should not be used as the sole basis for financial decisions.
