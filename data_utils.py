"""
Shared data-loading & cleaning utilities.

This mirrors EXACTLY the cleaning steps performed in Credit_Score.ipynb /
test_model.py, so that the numbers shown on the "Data Insight" page and the
values consumed by the trained model are always consistent with each other.
"""

import numpy as np
import pandas as pd
import streamlit as st

RAW_CSV_PATH = "credit__score.csv"

DROP_COLS = ["ID", "Customer_ID", "SSN", "Name"]

CLEAN_NUMERIC_UNDERSCORE_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt"
]

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August"]


def _clean_underscore_number(x):
    if pd.isna(x):
        return x
    return float(str(x).replace("_", "").replace("-", ""))


def _clean_occupation(x):
    if pd.isna(x):
        return np.nan
    if "_______" in str(x):
        return np.nan
    return x


def _clean_loan_types(x):
    if pd.isna(x):
        return x
    loans = str(x).replace(" and ", ",").split(",")
    loans = [loan.strip() for loan in loans]
    loans = list(set(loans))
    loans.sort()
    return ", ".join(loans)


def _clean_changed_limit(x):
    if pd.isna(x):
        return x
    if str(x)[0] == "_":
        return np.nan
    return float(x)


def _clean_credit_history_age(x):
    if pd.isna(x):
        return x
    parts = str(x).replace("Years", "").replace("Months", "").replace("and", "").split()
    return int(parts[0]) * 12 + int(parts[1])


def _clean_investment(x):
    if pd.isna(x):
        return np.nan
    return float(str(x).replace("_", ""))


def _clean_balance(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("_", "")
    try:
        value = float(x)
        return value if value >= 0 else np.nan
    except ValueError:
        return np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline from the notebook to a raw dataframe."""
    df = df.copy()

    existing_drop = [c for c in DROP_COLS if c in df.columns]
    if existing_drop:
        df = df.drop(existing_drop, axis=1)

    for col in CLEAN_NUMERIC_UNDERSCORE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_clean_underscore_number)

    if "Occupation" in df.columns:
        df["Occupation"] = df["Occupation"].apply(_clean_occupation)

    if "Type_of_Loan" in df.columns:
        df["Type_of_Loan"] = df["Type_of_Loan"].apply(_clean_loan_types)

    if "Changed_Credit_Limit" in df.columns:
        df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].apply(_clean_changed_limit)

    if "Credit_Mix" in df.columns:
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)

    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(_clean_credit_history_age)

    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

    if "Amount_invested_monthly" in df.columns:
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].apply(_clean_investment)

    if "Payment_Behaviour" in df.columns:
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

    if "Monthly_Balance" in df.columns:
        df["Monthly_Balance"] = df["Monthly_Balance"].apply(_clean_balance)

    # Outlier handling — identical thresholds to the notebook
    if "Age" in df.columns:
        df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
    if "Num_Bank_Accounts" in df.columns:
        df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
    if "Num_Credit_Card" in df.columns:
        df.loc[df["Num_Credit_Card"] > 20, "Num_Credit_Card"] = np.nan
    if "Interest_Rate" in df.columns:
        df.loc[df["Interest_Rate"] > 50, "Interest_Rate"] = np.nan
    if "Num_of_Loan" in df.columns:
        df.loc[df["Num_of_Loan"] > 20, "Num_of_Loan"] = np.nan
    if "Delay_from_due_date" in df.columns:
        df.loc[df["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
    if "Num_of_Delayed_Payment" in df.columns:
        df.loc[df["Num_of_Delayed_Payment"] > 100, "Num_of_Delayed_Payment"] = np.nan
    if "Num_Credit_Inquiries" in df.columns:
        df.loc[df["Num_Credit_Inquiries"] > 20, "Num_Credit_Inquiries"] = np.nan
    if "Amount_invested_monthly" in df.columns:
        df.loc[df["Amount_invested_monthly"] > 5000, "Amount_invested_monthly"] = np.nan
    if "Total_EMI_per_month" in df.columns:
        df.loc[df["Total_EMI_per_month"] > 2000, "Total_EMI_per_month"] = np.nan
    if "Annual_Income" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        mask = df["Annual_Income"] > 250000
        df.loc[mask, "Annual_Income"] = df.loc[mask, "Monthly_Inhand_Salary"] * 12

    missing_values = ["Not Specified", "Unknown", "_", "--"]
    if "Type_of_Loan" in df.columns:
        df["Type_of_Loan"] = df["Type_of_Loan"].replace(missing_values, np.nan)

    return df


@st.cache_data(show_spinner=False)
def load_and_clean_data(path: str = RAW_CSV_PATH):
    """Load the raw CSV and return the cleaned dataframe. Returns None if missing."""
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        return None
    return clean_dataframe(raw)
