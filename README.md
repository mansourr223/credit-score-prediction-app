# 💳 Credit Score Prediction App

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning web app that classifies a customer's credit
score into **Good**, **Standard**, or **Poor** based on their financial
behaviour, credit history, loans, and payment patterns — built with a
SMOTE-balanced Random Forest model and served through a custom-themed
Streamlit dashboard.

**🔗 Live demo:** _add your Streamlit Cloud link here after deploying_

---

## 📸 Screenshots

| About | Data Insight | Model Performance | Prediction |
|---|---|---|---|
| _add screenshot_ | _add screenshot_ | _add screenshot_ | _add screenshot_ |

---

## ✨ Features

- **📊 Data Insight** — live EDA dashboard (class balance, correlations,
  income/age distributions, debt-by-score, occupation & payment-behaviour
  breakdowns) computed directly from the dataset at runtime
- **🧪 Model Performance** — comparison of 4 algorithms (Gaussian Naive
  Bayes, Decision Tree, Random Forest, XGBoost) benchmarked with 5-fold
  cross-validation, plus a train-vs-test overfitting check and a
  hyperparameter-tuning experiment writeup
- **🔮 Prediction** — score a single customer through a form, or batch-score
  an uploaded CSV and download the results with per-class confidence scores
- Custom dark, fintech-style UI theme (no default Streamlit look)

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn (SMOTE)` ·
`XGBoost` · `Plotly` · `Streamlit`

## 📁 Project Structure

```
credit-score-prediction-app/
├── app.py                  # Main app — page router + all 4 pages
├── theme.py                # Shared dark theme CSS + UI helpers
├── data_utils.py           # Data cleaning pipeline (matches training notebook)
├── model_results.py        # Model comparison numbers from cross-validation
├── requirements.txt        # Python dependencies
├── random_forest_model.pkl # Trained production model (add your own)
├── credit__score.csv       # Dataset used by the Data Insight page (add your own)
└── sample_data.csv         # Optional sample rows for quick demo predictions
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/credit-score-prediction-app.git
cd credit-score-prediction-app
pip install -r requirements.txt
streamlit run app.py
```

> Make sure `random_forest_model.pkl` and `credit__score.csv` are present in
> the project root — see `README.md` inside the app folder for details on
> where these come from.

## 📈 Model Performance Summary

| Model | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---|---|---|---|
| Gaussian NB | 60.34% | 65.71% | 55.97% | 55.84% |
| Decision Tree | 61.62% | 62.38% | 61.98% | 64.35% |
| **Random Forest ✅** | **72.36%** | **75.19%** | **73.45%** | **74.87%** |
| XGBoost | 72.71% | 72.75% | 72.71% | — |

Random Forest (default hyperparameters, SMOTE-balanced) was selected as the
production model — full comparison and reasoning are in the app's **Model
Performance** page.

## 🗃️ Dataset

Customer credit data with financial, loan, and payment-behaviour features
across multiple months, cleaned to handle corrupted numeric strings, mixed
types, and outliers before modeling.

## 📄 License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name** — Data Scientist
