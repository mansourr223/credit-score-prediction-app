import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from theme import (
    inject_global_css, kpi_card, section_title, style_fig,
    ACCENT_1, ACCENT_2, CLASS_COLORS, PLOTLY_COLORWAY, TEXT_MUTED, CARD_BORDER
)
from data_utils import load_and_clean_data
from model_results import MODEL_COMPARISON, FINAL_MODEL, TUNING_EXPERIMENT, CLASS_DISTRIBUTION

# ============================================================
# APP AUTHOR — edit this to your name / title before deploying
# ============================================================
APP_AUTHOR = "Mahmoud Mansour"
APP_AUTHOR_TITLE = "Data Scientist"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Score Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# ============================================================
# MODEL / DATA COLUMN CONFIG (must match the training pipeline)
# ============================================================
pip1_cols = [
    "Age", "Monthly_Inhand_Salary", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Credit_History_Age", "Amount_invested_monthly", "Total_EMI_per_month"
]
pip2_cols = [
    "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Delay_from_due_date", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Monthly_Balance"
]
pip3_cols = ["Occupation", "Type_of_Loan", "Payment_Behaviour", "Month"]
pip4_cols = ["Credit_Mix", "Payment_of_Min_Amount"]
pip5_cols = ["Outstanding_Debt", "Credit_Utilization_Ratio"]
ALL_COLS = pip1_cols + pip2_cols + pip3_cols + pip4_cols + pip5_cols

occupation_options = [
    "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer",
    "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant",
    "Musician", "Mechanic", "Writer", "Architect"
]
loan_types = [
    "Auto Loan", "Personal Loan", "Credit-Builder Loan", "Home Equity Loan",
    "Mortgage Loan", "Payday Loan", "Student Loan", "Debt Consolidation Loan",
]
payment_behaviour_options = [
    "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
]
months = ["January", "February", "March", "April", "May", "June", "July", "August"]
credit_mix_options = ["Good", "Standard", "Bad"]
payment_of_min = ["Yes", "No"]


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("random_forest_model.pkl")


@st.cache_data(show_spinner=False)
def load_sample_data():
    try:
        return pd.read_csv("sample_data.csv")
    except FileNotFoundError:
        return None


# ============================================================
# SIDEBAR — brand + nav
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,{ACCENT_1},{ACCENT_2});
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem;">💳</div>
            <div style="font-family:'Poppins',sans-serif;font-weight:700;font-size:1.05rem;">Credit Score AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='muted' style='font-size:0.8rem;margin-bottom:1.2rem;'>Financial risk classification</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  About", "📊  Data Insight", "🧪  Model Performance", "🔮  Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ℹ️ Training data only covers **January–August**, so the Month field is limited to that range.")
    st.markdown(f"<div class='muted' style='font-size:0.75rem;margin-top:1rem;'>Designed &amp; developed by<br><strong style='color:#e7eaf0;'>{APP_AUTHOR}</strong></div>", unsafe_allow_html=True)

page = page.split("  ", 1)[1]

# ============================================================
# PAGE: ABOUT
# ============================================================
def page_about():
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="badge-pill">MACHINE LEARNING PROJECT</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <h1 style="font-size:2.6rem;line-height:1.15;margin-bottom:0.3rem;">
                Credit Score<br><span class="grad-text">Prediction System</span>
            </h1>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p class="muted" style="font-size:1.05rem;max-width:640px;margin-top:0.8rem;">
            An end-to-end machine learning application that classifies a customer's
            credit score into <strong style="color:#e7eaf0;">Good</strong>,
            <strong style="color:#e7eaf0;">Standard</strong>, or
            <strong style="color:#e7eaf0;">Poor</strong> using their financial
            behaviour, credit history, loans and payment patterns.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='muted' style='margin-top:1rem;'>Designed &amp; Developed by "
            f"<strong style='color:#e7eaf0;'>{APP_AUTHOR}</strong> · {APP_AUTHOR_TITLE}</p>",
            unsafe_allow_html=True,
        )
        go_to = st.button("🔮  Start Prediction  →")
        if go_to:
            st.info("Use the sidebar and select **🔮 Prediction** to try it out.")

    with right:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;padding:2.2rem 1.5rem;">
                <div style="width:150px;height:150px;border-radius:50%;margin:0 auto 1.1rem auto;
                            background:radial-gradient(circle at 35% 30%, {ACCENT_1}, {ACCENT_2});
                            box-shadow:0 0 60px rgba(239,68,68,0.35);
                            display:flex;align-items:center;justify-content:center;font-size:3.4rem;">💳</div>
                <h4 style="margin-bottom:0.2rem;">Credit Risk Intelligence</h4>
                <div class="muted" style="font-size:0.85rem;">Accurate • Explainable • Data Driven</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:2.2rem;'></div>", unsafe_allow_html=True)
    section_title("📌 Project Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Dataset Size", "100,000", "customer-month records")
    with c2:
        kpi_card("Features", "28 → 23", "used after cleaning")
    with c3:
        kpi_card("Target Classes", "3", "Good · Standard · Poor")
    with c4:
        kpi_card("Best Model", "Random Forest", f"{FINAL_MODEL['test_accuracy']:.1f}% test accuracy")

    section_title("🧭 How this app is built")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="card">
            <h4>1 · Data Cleaning &amp; EDA</h4>
            <p class="muted">Messy raw fields (underscores, corrupted values, mixed types,
            free-text ages) were parsed and cleaned, outliers capped, and the
            relationships between features explored — see <strong>Data Insight</strong>.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="card">
            <h4>2 · Model Comparison</h4>
            <p class="muted">Four algorithms were benchmarked with 5-fold cross-validation
            inside SMOTE-balanced pipelines: Naive Bayes, Decision Tree, Random Forest
            and XGBoost — see <strong>Model Performance</strong>.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="card">
            <h4>3 · Live Prediction</h4>
            <p class="muted">The tuned production pipeline (preprocessing + SMOTE + Random
            Forest) is served here for single-customer or batch CSV scoring — see
            <strong>Prediction</strong>.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    section_title("🛠️ Tech Stack")
    chips = ["Python", "Pandas", "scikit-learn", "imbalanced-learn (SMOTE)",
             "XGBoost", "category-encoders", "Plotly", "Streamlit"]
    st.markdown("".join(f'<span class="tech-chip">{c}</span>' for c in chips), unsafe_allow_html=True)


# ============================================================
# PAGE: DATA INSIGHT
# ============================================================
def page_data_insight():
    st.markdown('<div class="badge-pill">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:2rem;">Data <span class="grad-text">Insights</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Computed live from <code>credit__score.csv</code> using the same cleaning pipeline as the model.</p>', unsafe_allow_html=True)

    df = load_and_clean_data()

    if df is None:
        st.warning(
            "⚠️ Couldn't find **credit__score.csv** next to this app. "
            "Place the dataset file in the same folder to unlock this dashboard, "
            "or upload it below."
        )
        uploaded = st.file_uploader("Upload credit__score.csv", type=["csv"])
        if uploaded is None:
            return
        from data_utils import clean_dataframe
        df = clean_dataframe(pd.read_csv(uploaded))

    # ---------- KPI row ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Records", f"{len(df):,}")
    with c2:
        kpi_card("Avg. Age", f"{df['Age'].mean():.0f} yrs")
    with c3:
        kpi_card("Avg. Annual Income", f"${df['Annual_Income'].mean():,.0f}")
    with c4:
        good_pct = (df["Credit_Score"] == "Good").mean() * 100 if "Credit_Score" in df.columns else None
        kpi_card("‘Good’ Score Rate", f"{good_pct:.1f}%" if good_pct is not None else "—")

    # ---------- Credit score distribution + Credit mix ----------
    section_title("🎯 Target Distribution")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        if "Credit_Score" in df.columns:
            vc = df["Credit_Score"].value_counts(normalize=True).mul(100).round(2)
            fig = go.Figure(go.Pie(
                labels=vc.index, values=vc.values, hole=0.55,
                marker=dict(colors=[CLASS_COLORS.get(k, ACCENT_1) for k in vc.index]),
                textinfo="label+percent",
            ))
            fig.update_layout(title="Credit Score Class Balance", showlegend=False)
            st.plotly_chart(style_fig(fig, height=340), use_container_width=True)
    with col2:
        if {"Credit_Mix", "Credit_Score"}.issubset(df.columns):
            ct = pd.crosstab(df["Credit_Mix"], df["Credit_Score"], normalize="index").mul(100)
            fig = go.Figure()
            for cls in ["Good", "Standard", "Poor"]:
                if cls in ct.columns:
                    fig.add_bar(name=cls, x=ct.index, y=ct[cls], marker_color=CLASS_COLORS[cls])
            fig.update_layout(barmode="stack", title="Credit Score by Credit Mix (%)",
                               yaxis_title="Share of customers (%)")
            st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    # ---------- Distributions ----------
    section_title("📈 Numerical Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age", nbins=40, color_discrete_sequence=[ACCENT_1])
        fig.update_layout(title="Age Distribution")
        st.plotly_chart(style_fig(fig, height=320), use_container_width=True)
    with col2:
        fig = px.histogram(df, x="Annual_Income", nbins=40, color_discrete_sequence=[ACCENT_2])
        fig.update_layout(title="Annual Income Distribution")
        st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "Outstanding_Debt" in df.columns and "Credit_Score" in df.columns:
            fig = px.box(df, x="Credit_Score", y="Outstanding_Debt",
                         color="Credit_Score", color_discrete_map=CLASS_COLORS,
                         category_orders={"Credit_Score": ["Good", "Standard", "Poor"]})
            fig.update_layout(title="Outstanding Debt by Credit Score", showlegend=False)
            st.plotly_chart(style_fig(fig, height=320), use_container_width=True)
    with col4:
        if "Num_of_Loan" in df.columns and "Outstanding_Debt" in df.columns:
            sample = df.sample(min(4000, len(df)), random_state=42)
            fig = px.scatter(sample, x="Num_of_Loan", y="Outstanding_Debt",
                              color="Credit_Score" if "Credit_Score" in df.columns else None,
                              color_discrete_map=CLASS_COLORS, opacity=0.5)
            fig.update_layout(title="Number of Loans vs. Outstanding Debt")
            st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

    # ---------- Correlation heatmap ----------
    section_title("🔗 Feature Correlation")
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Correlation Matrix — Numerical Features")
        st.plotly_chart(style_fig(fig, height=560), use_container_width=True)

    # ---------- Categorical breakdowns ----------
    section_title("🗂️ Categorical Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        if "Occupation" in df.columns:
            vc = df["Occupation"].value_counts().sort_values(ascending=True)
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", color_discrete_sequence=[ACCENT_1])
            fig.update_layout(title="Customers by Occupation", xaxis_title="Count", yaxis_title="")
            st.plotly_chart(style_fig(fig, height=460), use_container_width=True)
    with col2:
        if "Payment_Behaviour" in df.columns:
            vc = df["Payment_Behaviour"].value_counts().sort_values(ascending=True)
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", color_discrete_sequence=[ACCENT_2])
            fig.update_layout(title="Payment Behaviour Breakdown", xaxis_title="Count", yaxis_title="")
            st.plotly_chart(style_fig(fig, height=460), use_container_width=True)

    # ---------- Trend across months ----------
    if "Month" in df.columns and "Amount_invested_monthly" in df.columns:
        section_title("🗓️ Monthly Trend")
        monthly = df.groupby("Month")["Amount_invested_monthly"].mean().reindex(months).reset_index()
        fig = px.line(monthly, x="Month", y="Amount_invested_monthly", markers=True,
                      color_discrete_sequence=[ACCENT_1])
        fig.update_layout(title="Average Monthly Investment by Month")
        st.plotly_chart(style_fig(fig, height=340), use_container_width=True)


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
def page_model_performance():
    st.markdown('<div class="badge-pill">EXPERIMENT TRACKING</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:2rem;">Model <span class="grad-text">Performance</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Four algorithms benchmarked with 5-fold cross-validation on SMOTE-balanced, preprocessed data. All scores are macro-averaged.</p>', unsafe_allow_html=True)

    comp = pd.DataFrame(MODEL_COMPARISON)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Champion Model", "Random Forest", "selected for production")
    with c2:
        kpi_card("Test Accuracy", f"{FINAL_MODEL['test_accuracy']:.2f}%")
    with c3:
        kpi_card("Test F1 (macro)", f"{FINAL_MODEL['test_f1_macro']:.2f}%")
    with c4:
        kpi_card("CV Folds", str(FINAL_MODEL["cv_folds"]))

    section_title("⚔️ Model Comparison — Test Set (macro-averaged)")
    metric_choice = st.radio(
        "Metric", ["F1 Score", "Precision", "Recall", "Accuracy"],
        horizontal=True, label_visibility="collapsed",
    )
    metric_map = {
        "F1 Score": "test_f1", "Precision": "test_precision",
        "Recall": "test_recall", "Accuracy": "test_accuracy",
    }
    mcol = metric_map[metric_choice]
    plot_df = comp.dropna(subset=[mcol]).sort_values(mcol, ascending=True)
    colors = [ACCENT_1 if m != "Random Forest" else ACCENT_2 for m in plot_df["model"]]
    fig = go.Figure(go.Bar(
        x=plot_df[mcol], y=plot_df["model"], orientation="h",
        marker_color=colors, text=plot_df[mcol].map(lambda v: f"{v:.2f}%"), textposition="outside",
    ))
    fig.update_layout(title=f"Test {metric_choice} by Model", xaxis_title=f"{metric_choice} (%)", yaxis_title="")
    st.plotly_chart(style_fig(fig, height=340), use_container_width=True)
    if metric_choice == "Accuracy":
        st.caption("XGBoost accuracy wasn't captured in this cross-validation run (only precision/recall/F1 were scored).")

    section_title("📉 Train vs. Test — Overfitting Check (F1 macro)")
    fig = go.Figure()
    fig.add_bar(name="Train F1", x=comp["model"], y=comp["train_f1"], marker_color="#3b4863")
    fig.add_bar(name="Test F1", x=comp["model"], y=comp["test_f1"], marker_color=ACCENT_1)
    fig.update_layout(barmode="group", title="Training vs. Test F1 Score", yaxis_title="F1 Score (%)")
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)
    st.caption(
        "Decision Tree and Random Forest reach 100% on training data — expected for "
        "fully-grown trees. What matters is the **test** score, where Random Forest "
        "and XGBoost clearly generalize best."
    )

    section_title("📋 Full Comparison Table")
    display_df = comp.copy()
    display_df["test_accuracy"] = display_df["test_accuracy"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
    for c in ["train_precision", "test_precision", "train_recall", "test_recall", "train_f1", "test_f1"]:
        display_df[c] = display_df[c].map(lambda v: f"{v:.2f}%")
    display_df = display_df.rename(columns={
        "model": "Model", "train_precision": "Train Prec.", "test_precision": "Test Prec.",
        "train_recall": "Train Rec.", "test_recall": "Test Rec.",
        "train_f1": "Train F1", "test_f1": "Test F1", "test_accuracy": "Test Acc.",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    section_title("🎛️ Hyperparameter Tuning Experiment")
    st.markdown(
        f"""
        <div class="card">
        <p class="muted" style="margin-bottom:0.6rem;">
        A <code>RandomizedSearchCV</code> was run over
        <code>{TUNING_EXPERIMENT['search_space']}</code>.
        </p>
        <p style="margin:0;">Best params: <strong>{TUNING_EXPERIMENT['best_params']}</strong>
        &nbsp;·&nbsp; Best CV F1 (macro): <strong>{TUNING_EXPERIMENT['best_cv_f1_macro']:.2f}%</strong>
        &nbsp;·&nbsp; Test F1 (macro): <strong>{TUNING_EXPERIMENT['test_f1_macro']:.2f}%</strong></p>
        <p class="muted" style="margin-top:0.8rem;">🧩 {TUNING_EXPERIMENT['conclusion']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("⚖️ Class Distribution in Training Data")
    dist = pd.Series(CLASS_DISTRIBUTION)
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[CLASS_COLORS[k] for k in dist.index],
        text=[f"{v:.1f}%" for v in dist.values], textposition="outside",
    ))
    fig.update_layout(title="Original Class Balance (before SMOTE)", yaxis_title="Share (%)")
    st.plotly_chart(style_fig(fig, height=320), use_container_width=True)
    st.caption("The training set is imbalanced (Standard 53% / Poor 29% / Good 18%), which is why SMOTE is applied inside the pipeline before fitting each model.")


# ============================================================
# PAGE: PREDICTION
# ============================================================
def page_prediction():
    st.markdown('<div class="badge-pill">LIVE INFERENCE</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:2rem;">Credit Score <span class="grad-text">Prediction</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Score a single customer manually, or upload a CSV for batch predictions.</p>', unsafe_allow_html=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure **random_forest_model.pkl** is in the same folder as this app.")
        st.stop()

    sample_data = load_sample_data()

    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Prediction"])

    # ---------------- TAB 1: MANUAL ----------------
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col2:
            if sample_data is not None and st.button("📋 Load Sample Customer", key="load_sample_btn"):
                st.session_state.load_sample = True

        if sample_data is not None and st.session_state.get("load_sample"):
            sample_idx = st.selectbox("Choose a sample customer:", range(len(sample_data)))
            sample_row = sample_data.iloc[sample_idx]
            st.session_state.load_sample = False
        else:
            sample_row = None
            sample_idx = "manual"

        with st.form("credit_score_form"):
            section_title("💰 Numerical Features")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                age = st.number_input("Age", min_value=18, max_value=100,
                                       value=int(sample_row["Age"]) if sample_row is not None else 35,
                                       key=f"age_{sample_idx}")
                monthly_salary = st.number_input("Monthly In-Hand Salary", min_value=0.0,
                                                  value=float(sample_row["Monthly_Inhand_Salary"]) if sample_row is not None else 3500.0,
                                                  step=100.0, key=f"salary_{sample_idx}")
                num_loans = st.number_input("Number of Loans", min_value=0, max_value=20,
                                             value=int(sample_row["Num_of_Loan"]) if sample_row is not None else 2,
                                             key=f"numloans_{sample_idx}")
            with col1b:
                num_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=20,
                                                        value=int(sample_row["Num_of_Delayed_Payment"]) if sample_row is not None else 1,
                                                        key=f"numdelayed_{sample_idx}")
                credit_history_age = st.number_input("Credit History Age (months)", min_value=0, max_value=600,
                                                       value=int(sample_row["Credit_History_Age"]) if sample_row is not None else 240,
                                                       key=f"credithist_{sample_idx}")
                amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0,
                                                    value=float(sample_row["Amount_invested_monthly"]) if sample_row is not None else 200.0,
                                                    step=50.0, key=f"invested_{sample_idx}")
            with col1c:
                total_emi = st.number_input("Total EMI Per Month", min_value=0.0,
                                             value=float(sample_row["Total_EMI_per_month"]) if sample_row is not None else 150.0,
                                             step=50.0, key=f"emi_{sample_idx}")

            section_title("📊 Additional Financial Metrics")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                annual_income = st.number_input("Annual Income", min_value=0.0,
                                                  value=float(sample_row["Annual_Income"]) if sample_row is not None else 50000.0,
                                                  step=1000.0, key=f"income_{sample_idx}")
                num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20,
                                                      value=int(sample_row["Num_Bank_Accounts"]) if sample_row is not None else 2,
                                                      key=f"bankacc_{sample_idx}")
                num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=20,
                                                     value=int(sample_row["Num_Credit_Card"]) if sample_row is not None else 3,
                                                     key=f"credcards_{sample_idx}")
            with col2b:
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0,
                                                  value=float(sample_row["Interest_Rate"]) if sample_row is not None else 5.5,
                                                  step=0.1, key=f"intrate_{sample_idx}")
                delay_from_due = st.number_input("Delay from Due Date (days)", min_value=0, max_value=365,
                                                   value=int(sample_row["Delay_from_due_date"]) if sample_row is not None else 2,
                                                   key=f"delaydue_{sample_idx}")
                changed_credit_limit = st.number_input("Changed Credit Limit", min_value=-100.0,
                                                          value=float(sample_row["Changed_Credit_Limit"]) if sample_row is not None else 5.5,
                                                          step=0.1, key=f"changedlimit_{sample_idx}")
            with col2c:
                num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=20,
                                                          value=int(sample_row["Num_Credit_Inquiries"]) if sample_row is not None else 2,
                                                          key=f"inquiries_{sample_idx}")
                monthly_balance = st.number_input("Monthly Balance", min_value=0.0,
                                                    value=float(sample_row["Monthly_Balance"]) if sample_row is not None else 2500.0,
                                                    step=100.0, key=f"balance_{sample_idx}")

            section_title("🗂️ Categorical Features")
            col3a, col3b, col3c = st.columns(3)
            with col3a:
                occupation = st.selectbox("Occupation", occupation_options,
                                           index=occupation_options.index(sample_row["Occupation"]) if sample_row is not None and sample_row["Occupation"] in occupation_options else 0,
                                           key=f"occ_{sample_idx}")
                month = st.selectbox("Month", months,
                                      index=months.index(sample_row["Month"]) if sample_row is not None and sample_row["Month"] in months else 0,
                                      key=f"month_{sample_idx}")
            with col3b:
                if sample_row is not None and pd.notna(sample_row["Type_of_Loan"]):
                    sample_loans = [l.strip() for l in str(sample_row["Type_of_Loan"]).split(",")]
                    safe_default = [l for l in sample_loans if l in loan_types] or ["Auto Loan"]
                else:
                    safe_default = ["Auto Loan"]
                selected_loans = st.multiselect("Type(s) of Loan", loan_types, default=safe_default, key=f"loans_{sample_idx}")
                type_of_loan = ", ".join(sorted(selected_loans)) if selected_loans else np.nan
            with col3c:
                payment_behaviour = st.selectbox(
                    "Payment Behaviour", payment_behaviour_options,
                    index=payment_behaviour_options.index(sample_row["Payment_Behaviour"]) if sample_row is not None and sample_row["Payment_Behaviour"] in payment_behaviour_options else 0,
                    key=f"paybehav_{sample_idx}")

            section_title("📈 Credit Profile")
            col4a, col4b, col4c = st.columns(3)
            with col4a:
                credit_mix = st.selectbox("Credit Mix", credit_mix_options,
                                           index=credit_mix_options.index(sample_row["Credit_Mix"]) if sample_row is not None and sample_row["Credit_Mix"] in credit_mix_options else 0,
                                           key=f"credmix_{sample_idx}")
            with col4b:
                payment_min_amount = st.selectbox("Payment of Minimum Amount", payment_of_min,
                                                    index=payment_of_min.index(sample_row["Payment_of_Min_Amount"]) if sample_row is not None and sample_row["Payment_of_Min_Amount"] in payment_of_min else 0,
                                                    key=f"payminamt_{sample_idx}")
            with col4c:
                outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0,
                                                     value=float(sample_row["Outstanding_Debt"]) if sample_row is not None else 800.0,
                                                     step=50.0, key=f"debt_{sample_idx}")
            credit_utilization = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0,
                                                   value=float(sample_row["Credit_Utilization_Ratio"]) if sample_row is not None else 35.5,
                                                   step=0.5, key=f"utilization_{sample_idx}")

            st.markdown("---")
            submit_button = st.form_submit_button("🔮 Predict Credit Score", use_container_width=True)

        if submit_button:
            try:
                input_data = pd.DataFrame({
                    "Age": [age], "Monthly_Inhand_Salary": [monthly_salary], "Num_of_Loan": [num_loans],
                    "Num_of_Delayed_Payment": [num_delayed_payment], "Credit_History_Age": [credit_history_age],
                    "Amount_invested_monthly": [amount_invested], "Total_EMI_per_month": [total_emi],
                    "Annual_Income": [annual_income], "Num_Bank_Accounts": [num_bank_accounts],
                    "Num_Credit_Card": [num_credit_cards], "Interest_Rate": [interest_rate],
                    "Delay_from_due_date": [delay_from_due], "Changed_Credit_Limit": [changed_credit_limit],
                    "Num_Credit_Inquiries": [num_credit_inquiries], "Monthly_Balance": [monthly_balance],
                    "Occupation": [occupation], "Type_of_Loan": [type_of_loan],
                    "Payment_Behaviour": [payment_behaviour], "Month": [month],
                    "Credit_Mix": [credit_mix], "Payment_of_Min_Amount": [payment_min_amount],
                    "Outstanding_Debt": [outstanding_debt], "Credit_Utilization_Ratio": [credit_utilization],
                })[ALL_COLS]

                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]

                section_title("🎯 Prediction Result")
                css_class = {"Good": "pred-good", "Standard": "pred-standard", "Poor": "pred-poor"}.get(prediction, "pred-standard")
                emoji = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}.get(prediction, "⚠️")
                st.markdown(f"""
                <div class="pred-box {css_class}">
                    <h3>{emoji} Credit Score: <strong>{prediction}</strong></h3>
                    <p class="muted">The customer is predicted to have a <strong style="color:#e7eaf0;">{prediction}</strong> credit score.</p>
                </div>
                """, unsafe_allow_html=True)

                section_title("📊 Confidence Breakdown")
                sorted_classes = sorted(model.classes_)
                cols = st.columns(3)
                for idx, cls in enumerate(sorted_classes):
                    conf_pct = probabilities[list(model.classes_).index(cls)] * 100
                    with cols[idx % 3]:
                        kpi_card(cls, f"{conf_pct:.1f}%")

                conf_df = pd.DataFrame({
                    "Class": sorted_classes,
                    "Confidence": [probabilities[list(model.classes_).index(cls)] * 100 for cls in sorted_classes],
                })
                fig = go.Figure(go.Bar(
                    x=conf_df["Class"], y=conf_df["Confidence"],
                    marker_color=[CLASS_COLORS.get(c, ACCENT_1) for c in conf_df["Class"]],
                    text=conf_df["Confidence"].map(lambda v: f"{v:.1f}%"), textposition="outside",
                ))
                fig.update_layout(title="Prediction Confidence by Class", yaxis_title="Confidence (%)")
                st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Make sure all fields are filled correctly and contain valid values.")

    # ---------------- TAB 2: BATCH ----------------
    with tab2:
        st.subheader("Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(batch_df)} records")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🔮 Predict for All Customers"):
                    try:
                        batch_df_ordered = batch_df[ALL_COLS]
                        predictions = model.predict(batch_df_ordered)
                        probabilities = model.predict_proba(batch_df_ordered)

                        results_df = batch_df.copy()
                        results_df["Predicted_Credit_Score"] = predictions
                        sorted_classes = sorted(model.classes_)
                        for cls in sorted_classes:
                            cls_idx = list(model.classes_).index(cls)
                            results_df[f"Confidence_{cls}"] = probabilities[:, cls_idx]

                        st.success("✅ Predictions completed!")
                        st.dataframe(results_df, use_container_width=True)

                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download Results (CSV)", data=csv,
                                            file_name="credit_score_predictions.csv", mime="text/csv")

                        section_title("📈 Summary Statistics")
                        summary_cols = st.columns(len(sorted_classes))
                        for idx, cls in enumerate(sorted_classes):
                            count = (predictions == cls).sum()
                            with summary_cols[idx]:
                                kpi_card(f"{cls} Count", str(count), f"{count/len(predictions)*100:.1f}% of batch")

                        vc = pd.Series(predictions).value_counts()
                        fig = go.Figure(go.Bar(
                            x=vc.index, y=vc.values,
                            marker_color=[CLASS_COLORS.get(c, ACCENT_1) for c in vc.index],
                        ))
                        fig.update_layout(title="Predicted Class Distribution — This Batch", yaxis_title="Count")
                        st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        st.info("Ensure the CSV has all required columns with correct names.")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")


# ============================================================
# ROUTER
# ============================================================
if page == "About":
    page_about()
elif page == "Data Insight":
    page_data_insight()
elif page == "Model Performance":
    page_model_performance()
elif page == "Prediction":
    page_prediction()

st.markdown("""
<div class="app-footer">
    💳 Credit Score Prediction System &nbsp;|&nbsp; Random Forest Classifier with SMOTE<br>
    Classes: Good, Standard, Poor
</div>
""", unsafe_allow_html=True)