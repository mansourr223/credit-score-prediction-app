import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load the CSV data
df = pd.read_csv("credit__score.csv")
df = df.drop(["ID", "Customer_ID", "SSN", "Name"], axis=1)

# ===== DATA CLEANING (from notebook) =====
# Clean columns that have string numbers with underscores
clean1_col = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt"]

def clean_col(x):
    if pd.isna(x):
        return x
    return float(str(x).replace("_", "").replace("-", ""))

for col in clean1_col:
    df[col] = df[col].apply(clean_col)

# Clean Occupation
def clean_occ(x):
    if pd.isna(x):
        return np.nan
    elif "_______" in str(x):
        return np.nan
    else:
        return x

df["Occupation"] = df["Occupation"].apply(clean_occ)

# Clean Type_of_Loan
def clean_loan(x):
    if pd.isna(x):
        return x
    loans = str(x).replace(" and ", ",").split(",")
    loans = [loan.strip() for loan in loans]
    loans = list(set(loans))
    loans.sort()
    return ", ".join(loans)

df["Type_of_Loan"] = df["Type_of_Loan"].apply(clean_loan)

# Clean Changed_Credit_Limit
def clean_changed_limits(x):
    if pd.isna(x):
        return x
    elif str(x)[0] == "_":
        return np.nan
    else:
        return float(x)

df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].apply(clean_changed_limits)

# Clean Credit_Mix
df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)

# Clean Credit_History_Age
def clean_credit_history(x):
    if pd.isna(x):
        return x
    else:
        x = str(x).replace("Years", "").replace("Months", "").replace("and", "").split()
        return int(x[0]) * 12 + int(x[1])

df["Credit_History_Age"] = df["Credit_History_Age"].apply(clean_credit_history)

# Clean Payment_of_Min_Amount
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

# Clean Amount_invested_monthly
def clean_investment(x):
    if pd.isna(x):
        return np.nan
    return float(str(x).replace("_", ""))

df["Amount_invested_monthly"] = df["Amount_invested_monthly"].apply(clean_investment)

# Clean Payment_Behaviour
df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

# Clean Monthly_Balance
def clean_balance(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("_", "")
    try:
        value = float(x)
        if value < 0:
            return np.nan
        return value
    except:
        return np.nan

df["Monthly_Balance"] = df["Monthly_Balance"].apply(clean_balance)

# Handle outliers
df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
df.loc[df["Num_Credit_Card"] > 100, "Num_Credit_Card"] = np.nan
df.loc[df["Interest_Rate"] > 100, "Interest_Rate"] = np.nan
df.loc[df["Num_of_Loan"] > 100, "Num_of_Loan"] = np.nan
df.loc[df["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
df.loc[df["Num_of_Delayed_Payment"] > 100, "Num_of_Delayed_Payment"] = np.nan
df.loc[df["Num_Credit_Inquiries"] > 100, "Num_Credit_Inquiries"] = np.nan
df.loc[df["Num_Credit_Card"] > 20, "Num_Credit_Card"] = np.nan
df.loc[df["Interest_Rate"] > 50, "Interest_Rate"] = np.nan
df.loc[df["Num_of_Loan"] > 20, "Num_of_Loan"] = np.nan
df.loc[df["Amount_invested_monthly"] > 5000, "Amount_invested_monthly"] = np.nan
df.loc[df["Num_Credit_Inquiries"] > 20, "Num_Credit_Inquiries"] = np.nan
df.loc[df["Total_EMI_per_month"] > 2000, "Total_EMI_per_month"] = np.nan
df.loc[df["Annual_Income"] > 250000, "Annual_Income"] = df.loc[df["Annual_Income"] > 250000, "Monthly_Inhand_Salary"] * 12

# Remove invalid Type_of_Loan values
missing_values = ["Not Specified", "Unknown", "_", "--"]
df["Type_of_Loan"] = df["Type_of_Loan"].replace(missing_values, np.nan)

# ===== END DATA CLEANING =====

# Prepare data like in the notebook
x = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = joblib.load("random_forest_model.pkl")

# Make predictions
y_pred = model.predict(x_test)

# Print classification report
print("=" * 70)
print("CLASSIFICATION REPORT (Per-Class Metrics)")
print("=" * 70)
print(classification_report(y_test, y_pred))

# Calculate macro-averaged metrics
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')

print("=" * 70)
print("MACRO-AVERAGED METRICS")
print("=" * 70)
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")

# Check per-class metrics
print("\n" + "=" * 70)
print("PER-CLASS PRECISION & RECALL (Threshold: >= 0.3)")
print("=" * 70)
classes = sorted(model.classes_)
all_pass = True
for cls in classes:
    mask = y_test == cls
    if mask.sum() == 0:
        continue
    prec = precision_score(y_test, y_pred, labels=[cls], average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, labels=[cls], average='binary', zero_division=0)
    prec_pass = "✓" if prec >= 0.3 else "✗"
    rec_pass = "✓" if rec >= 0.3 else "✗"
    print(f"{cls:12s} - Precision: {prec:.4f} {prec_pass} | Recall: {rec:.4f} {rec_pass}")
    if prec < 0.3 or rec < 0.3:
        all_pass = False

print("\n" + "=" * 70)
if all_pass:
    print("✓ ALL CLASSES MEET THE 0.3 THRESHOLD FOR PRECISION & RECALL")
else:
    print("✗ SOME CLASSES DO NOT MEET THE 0.3 THRESHOLD")
print("=" * 70)