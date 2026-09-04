"""Dark, fintech-style theme shared by every page of the app."""

import streamlit as st

# Brand palette -------------------------------------------------------------
BG = "#0a0e17"
BG_SOFT = "#0d1220"
CARD = "#131a2b"
CARD_BORDER = "#232b3f"
TEXT = "#e7eaf0"
TEXT_MUTED = "#8b93a7"
ACCENT_1 = "#fb923c"   # orange
ACCENT_2 = "#ef4444"   # red
ACCENT_GRADIENT = f"linear-gradient(90deg, {ACCENT_1} 0%, {ACCENT_2} 100%)"
GOOD = "#22c55e"
STANDARD = "#f59e0b"
POOR = "#ef4444"

CLASS_COLORS = {"Good": GOOD, "Standard": STANDARD, "Poor": POOR}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORWAY = [ACCENT_1, "#60a5fa", "#a78bfa", ACCENT_2, "#34d399", "#f472b6"]


def inject_global_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at 15% 0%, #101728 0%, {BG} 45%) fixed;
            color: {TEXT};
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        section[data-testid="stSidebar"] {{
            background: {BG_SOFT};
            border-right: 1px solid {CARD_BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

        h1, h2, h3, h4 {{
            font-family: 'Poppins', sans-serif;
            color: {TEXT};
        }}
        p, span, label, li {{ color: {TEXT}; }}
        .muted {{ color: {TEXT_MUTED} !important; }}

        /* ---- Sidebar nav (radio styled as nav links) ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 2px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 0.55rem 0.8rem;
            border-radius: 10px;
            width: 100%;
            transition: background 0.15s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: rgba(251, 146, 60, 0.08);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
            font-size: 0.95rem;
            font-weight: 500;
            color: {TEXT_MUTED};
        }}

        /* ---- Badge pill ---- */
        .badge-pill {{
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(251, 146, 60, 0.5);
            background: rgba(251, 146, 60, 0.08);
            color: {ACCENT_1};
            font-weight: 600;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1.1rem;
        }}

        /* ---- Gradient headline text ---- */
        .grad-text {{
            background: {ACCENT_GRADIENT};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ---- Generic card ---- */
        .card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
        }}
        .card h4 {{ margin-top: 0; }}

        /* ---- KPI stat card ---- */
        .kpi-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            height: 100%;
        }}
        .kpi-label {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .kpi-value {{
            font-family: 'Poppins', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: {TEXT};
            margin-top: 0.15rem;
        }}
        .kpi-sub {{ color: {TEXT_MUTED}; font-size: 0.8rem; margin-top: 0.1rem; }}

        /* ---- Section title ---- */
        .section-title {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            color: {TEXT};
            margin: 1.4rem 0 0.6rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid {CARD_BORDER};
        }}

        /* ---- Prediction result boxes ---- */
        .pred-box {{
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            margin: 1rem 0;
        }}
        .pred-box h3 {{ margin: 0 0 0.3rem 0; }}
        .pred-good {{ background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34,197,94,0.35); }}
        .pred-standard {{ background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245,158,11,0.35); }}
        .pred-poor {{ background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239,68,68,0.35); }}

        /* ---- Metric widgets ---- */
        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}

        /* ---- Buttons ---- */
        .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {{
            background: {ACCENT_GRADIENT};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: 0.2s;
        }}
        .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            font-weight: 600;
            color: {TEXT_MUTED};
        }}

        /* ---- Dataframe / table container ---- */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {CARD_BORDER};
            border-radius: 12px;
            overflow: hidden;
        }}

        .app-footer {{
            text-align: center;
            color: {TEXT_MUTED};
            font-size: 0.8rem;
            padding: 2rem 0 0.5rem 0;
        }}

        .tech-chip {{
            display: inline-block;
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            color: {TEXT_MUTED};
            border-radius: 8px;
            padding: 0.3rem 0.7rem;
            font-size: 0.78rem;
            margin: 0.15rem;
        }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=None):
    """Apply the shared dark plotly styling to a figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if height:
        fig.update_layout(height=height)
    return fig
