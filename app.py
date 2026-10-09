"""
app.py
======
Streamlit front-end for the Water Quality Prediction Agent.

Pages:
  Test Water  — Enter water parameters and get a safe/unsafe prediction.
  Dashboard   — History of all tests, charts and alert banners.
  Model Info  — Metrics, confusion matrix, feature importance.

Run:
    streamlit run app.py
"""

import os
import sqlite3
import pathlib
import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import streamlit as st

from config  import PARAM_CONFIG, FEATURE_COLUMNS, DISCLAIMER, HISTORY_DB_PATH, \
                     MODEL_PATH, CM_IMAGE_PATH, FI_IMAGE_PATH, METRICS_PATH
from advisor import check_parameters, get_ai_advice, build_rule_based_summary

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaSafe — Water Quality Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    :root {
        --bg:          #F5F4F0;
        --bg-card:     #FFFFFF;
        --bg-sidebar:  #1C2B2A;

        --teal:        #2A7C6F;
        --teal-dark:   #1B5C52;
        --teal-light:  #EAF4F2;
        --teal-mid:    #3D9E8E;

        --amber:       #D97706;
        --amber-light: #FEF3C7;
        --red:         #C0392B;
        --red-light:   #FDECEA;
        --green:       #1E7B4B;
        --green-light: #E8F5EE;

        --text:        #1A1A1A;
        --text-mid:    #4A5568;
        --text-soft:   #718096;
        --border:      #E2E8E4;

        --radius:      10px;
        --shadow-sm:   0 1px 4px rgba(0,0,0,0.07);
        --shadow-md:   0 3px 12px rgba(0,0,0,0.09);
        --shadow-lg:   0 6px 24px rgba(0,0,0,0.10);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text) !important;
    }

    /* ── Page background ── */
    .stApp {
        background-color: var(--bg) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5D1 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 6px 0 !important;
        color: #CBD5D1 !important;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #FFFFFF !important;
    }

    /* ── Sidebar Guide & Disclaimer ── */
    .sidebar-guide {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 1.2rem 0.8rem 0.6rem;
    }
    .sidebar-guide-title {
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: rgba(203, 213, 209, 0.6) !important;
        margin: 0 0 0.75rem 0;
    }
    .guide-step {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        margin-bottom: 0.6rem;
    }
    .guide-step:last-child {
        margin-bottom: 0;
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(42, 124, 111, 0.5);
        border: 1px solid rgba(61, 158, 142, 0.6);
        color: #A7F3D0 !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-text {
        font-size: 0.8rem !important;
        color: rgba(203, 213, 209, 0.9) !important;
        line-height: 1.4;
    }
    .step-text b {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    .sidebar-disclaimer {
        margin: 0.6rem 0.8rem 1.2rem;
        padding: 0.7rem 0.85rem;
        background: rgba(217, 119, 6, 0.08);
        border: 1px solid rgba(217, 119, 6, 0.22);
        border-radius: 7px;
    }
    .sidebar-disclaimer p {
        font-size: 0.72rem !important;
        color: rgba(254, 243, 199, 0.75) !important;
        line-height: 1.45;
        margin: 0;
    }

    /* ── Top header bar ── */
    .site-header {
        background-color: var(--teal);
        padding: 1.4rem 2rem 1.4rem;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        border-bottom: 3px solid var(--teal-dark);
    }
    .site-header-logo {
        font-size: 1.9rem;
        line-height: 1;
    }
    .site-header-text h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.55rem;
        color: #FFFFFF !important;
        margin: 0;
        line-height: 1.2;
    }
    .site-header-text p {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.75) !important;
        margin: 0;
        font-weight: 400;
    }
    .site-header-badge {
        margin-left: auto;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 99px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }

    /* ── Cards ── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--teal-dark) !important;
        margin: 0 0 0.3rem 0;
    }
    .card-desc {
        font-size: 0.85rem;
        color: var(--text-soft) !important;
        margin: 0;
    }

    /* ── Section headings ── */
    .sec-head {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--teal) !important;
        margin: 1.6rem 0 0.7rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sec-head::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── Result boxes ── */
    .result-safe {
        background: var(--green-light);
        border: 1.5px solid #86EFAC;
        border-left: 5px solid var(--green);
        border-radius: var(--radius);
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .result-unsafe {
        background: var(--red-light);
        border: 1.5px solid #FCA5A5;
        border-left: 5px solid var(--red);
        border-radius: var(--radius);
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .result-prob {
        font-size: 0.88rem;
        color: var(--text-mid) !important;
        margin: 0;
    }

    /* ── Alert banner ── */
    .alert-danger {
        background: #FFF5F5;
        border: 1px solid #FEB2B2;
        border-left: 4px solid var(--red);
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #7B1A1A !important;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: var(--amber-light);
        border: 1px solid #FCD34D;
        border-left: 4px solid var(--amber);
        border-radius: var(--radius);
        padding: 0.8rem 1.1rem;
        font-size: 0.82rem;
        color: #78350F !important;
        margin: 1rem 0 0;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="metric-container"] label {
        color: var(--text-soft) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: var(--teal) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        letter-spacing: 0.2px !important;
        transition: background-color 0.18s ease, box-shadow 0.18s ease !important;
        box-shadow: 0 1px 4px rgba(42,124,111,0.25) !important;
    }
    .stButton > button:hover {
        background-color: var(--teal-dark) !important;
        box-shadow: 0 3px 10px rgba(42,124,111,0.35) !important;
    }

    /* ── Number inputs ── */
    div[data-testid="stNumberInput"] input {
        background: #FAFAFA !important;
        border: 1px solid var(--border) !important;
        border-radius: 7px !important;
        color: var(--text) !important;
        font-size: 0.93rem !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(42,124,111,0.12) !important;
    }
    div[data-testid="stNumberInput"] label {
        color: var(--text-mid) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 2px solid var(--border) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-soft) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        border-radius: 0 !important;
        padding: 8px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--teal) !important;
        border-bottom: 2px solid var(--teal) !important;
        font-weight: 700 !important;
        background: transparent !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #CBD5D1; border-radius: 4px; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

    /* ── Header & controls ── */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    header { background: transparent !important; }
    [data-testid="stHeader"] {
        background: transparent !important;
        z-index: 99 !important;
    }
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        color: var(--teal) !important;
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        margin: 0.5rem 0 0 0.5rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
        transition: background-color 0.15s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: var(--teal-light) !important;
        color: var(--teal-dark) !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #CBD5D1 !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        color: #FFFFFF !important;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1180px !important;
    }
    code {
        background: var(--teal-light) !important;
        color: var(--teal-dark) !important;
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.83em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Database helpers ────────────────────────────────────────────────────────
DB_PATH = pathlib.Path(HISTORY_DB_PATH)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            ph          REAL, Hardness REAL, Solids REAL, Chloramines REAL,
            Sulfate     REAL, Conductivity REAL, Organic_carbon REAL,
            Trihalomethanes REAL, Turbidity REAL,
            prediction  INTEGER,
            probability REAL
        )"""
    )
    conn.commit()
    return conn


def save_test(values: dict, prediction: int, probability: float):
    conn = _get_conn()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """INSERT INTO history
           (timestamp,ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,
            Organic_carbon,Trihalomethanes,Turbidity,prediction,probability)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ts,
         values["ph"], values["Hardness"], values["Solids"], values["Chloramines"],
         values["Sulfate"], values["Conductivity"], values["Organic_carbon"],
         values["Trihalomethanes"], values["Turbidity"],
         prediction, probability),
    )
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df


# ── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    mp = pathlib.Path(MODEL_PATH)
    if not mp.exists():
        return None
    return joblib.load(mp)


# ── Plotly chart theme (light) ───────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAFA",
    font={"color": "#1A1A1A", "family": "DM Sans"},
    margin=dict(l=12, r=12, t=40, b=12),
)
GRID = dict(gridcolor="#E2E8E4", zerolinecolor="#E2E8E4")


# ── Header ──────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        """
        <div class="site-header">
            <div class="site-header-text">
                <h1>AquaSafe</h1>
                <p>Water Quality Prediction Agent</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Navigation ───────────────────────────────────────────────────────────────
def render_nav() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:1.6rem 1rem 0.8rem;">
                <p style="font-family:'DM Serif Display',serif;font-size:1.15rem;
                   color:#FFFFFF !important;margin:0;letter-spacing:0.2px;">AquaSafe</p>
                <p style="font-size:0.75rem;color:rgba(203,213,209,0.7) !important;margin:0;">
                    Water Quality Agent
                </p>
            </div>
            <hr style="border-color:rgba(255,255,255,0.1);margin:0 1rem 0.8rem;">
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            ["Test Water", "Dashboard", "Model Info"],
            label_visibility="hidden",
        )

        st.markdown(
            """
            <div class="sidebar-guide">
                <div class="sidebar-guide-title">How to use</div>
                <div class="guide-step">
                    <span class="step-badge">1</span>
                    <span class="step-text">Enter water parameters</span>
                </div>
                <div class="guide-step">
                    <span class="step-badge">2</span>
                    <span class="step-text">Click <b>Check Water</b></span>
                </div>
                <div class="guide-step">
                    <span class="step-badge">3</span>
                    <span class="step-text">Review result & advice</span>
                </div>
                <div class="guide-step">
                    <span class="step-badge">4</span>
                    <span class="step-text">Track history in Dashboard</span>
                </div>
            </div>
            <div class="sidebar-disclaimer">
                <p>For educational use only.<br>Not a substitute for certified lab testing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TEST WATER
# ══════════════════════════════════════════════════════════════════════════════
def page_test_water():
    model = load_model()
    if model is None:
        st.error("Model not found. Please run `python train.py` first.")
        return

    # ── Page intro ──
    st.markdown(
        """
        <div style="padding:1.4rem 0 0.4rem;">
            <h2 style="font-family:'DM Serif Display',serif;font-size:1.5rem;
               color:#1C2B2A;margin:0 0 0.3rem;">Water Quality Test</h2>
            <p style="font-size:0.88rem;color:#718096;margin:0;">
                Enter your sample's physicochemical parameters and press <b>Check Water</b> 
                to get a safety prediction and remediation advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Presets ──
    st.markdown('<p class="sec-head">Quick Presets</p>', unsafe_allow_html=True)

    PRESETS = {
        "Clean Well Water": {
            "ph": 7.2, "Hardness": 180.0, "Solids": 320.0, "Chloramines": 2.5,
            "Sulfate": 140.0, "Conductivity": 280.0, "Organic_carbon": 1.2,
            "Trihalomethanes": 25.0, "Turbidity": 0.5,
        },
        "Polluted Runoff": {
            "ph": 5.2, "Hardness": 620.0, "Solids": 38000.0, "Chloramines": 9.5,
            "Sulfate": 480.0, "Conductivity": 1800.0, "Organic_carbon": 22.0,
            "Trihalomethanes": 110.0, "Turbidity": 8.5,
        },
        "Borderline Sample": {
            "ph": 7.0, "Hardness": 300.0, "Solids": 8000.0, "Chloramines": 4.2,
            "Sulfate": 260.0, "Conductivity": 420.0, "Organic_carbon": 3.5,
            "Trihalomethanes": 82.0, "Turbidity": 2.8,
        },
    }

    def apply_preset(name: str):
        st.session_state.preset = name
        for p, v in PRESETS[name].items():
            st.session_state[f"inp_{p}"] = float(v)

    cp1, cp2, cp3, _ = st.columns([1, 1, 1, 2])
    with cp1:
        if st.button("Clean Well Water", width='stretch'):
            apply_preset("Clean Well Water")
            st.rerun()
    with cp2:
        if st.button("Polluted Runoff", width='stretch'):
            apply_preset("Polluted Runoff")
            st.rerun()
    with cp3:
        if st.button("Borderline Sample", width='stretch'):
            apply_preset("Borderline Sample")
            st.rerun()

    # Initialize default inputs in session state to safe defaults if not yet set
    for param, cfg in PARAM_CONFIG.items():
        if f"inp_{param}" not in st.session_state:
            st.session_state[f"inp_{param}"] = float(cfg["default"])

    # ── Parameter inputs ──
    st.markdown('<p class="sec-head">Water Parameters</p>', unsafe_allow_html=True)

    input_values = {}
    params_list  = list(PARAM_CONFIG.items())
    rows         = [params_list[i:i+3] for i in range(0, len(params_list), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, (param, cfg) in zip(cols, row):
            with col:
                val = st.number_input(
                    label=f"{cfg['display_name']} ({cfg['unit']})",
                    min_value=float(cfg["min_val"]),
                    max_value=float(cfg["max_val"]),
                    step=float(cfg["step"]),
                    help=cfg["description"],
                    key=f"inp_{param}",
                )
                input_values[param] = val

    # ── Submit ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, _ = st.columns([1, 3])
    with c1:
        check_clicked = st.button("Check Water Quality", width='stretch')

    if not check_clicked:
        st.markdown(
            f'<div class="disclaimer">{DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Prediction ──
    X_input = np.array([[input_values[c] for c in FEATURE_COLUMNS]])
    with st.spinner("Analysing sample..."):
        prediction = int(model.predict(X_input)[0])

    safe_prob   = float(model.predict_proba(X_input)[0][1])
    unsafe_prob = float(model.predict_proba(X_input)[0][0])
    save_test(input_values, prediction, safe_prob)

    # ── Result banner ──
    st.markdown('<p class="sec-head">Result</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-safe">
                <p class="result-label" style="color:#1E7B4B;">Safe to Drink</p>
                <p class="result-prob">
                    Model confidence: <b>{safe_prob:.1%} safe</b> &nbsp;·&nbsp;
                    Unsafe probability: {unsafe_prob:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-unsafe">
                <p class="result-label" style="color:#C0392B;">Unsafe — Do Not Drink</p>
                <p class="result-prob">
                    Model confidence: <b>{unsafe_prob:.1%} unsafe</b> &nbsp;·&nbsp;
                    Safe probability: {safe_prob:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Gauge + Parameter table side by side ──
    col_g, col_t = st.columns([1, 1])

    with col_g:
        bar_color = "#1E7B4B" if prediction == 1 else "#C0392B"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safe_prob * 100,
            title={"text": "Safety Score (%)", "font": {"size": 13, "color": "#4A5568"}},
            number={"suffix": "%", "font": {"size": 30, "color": "#1A1A1A"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#718096",
                         "tickfont": {"size": 10, "color": "#718096"}},
                "bar":  {"color": bar_color, "thickness": 0.25},
                "steps": [
                    {"range": [0, 40],   "color": "#FEE2E2"},
                    {"range": [40, 60],  "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#4A5568", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
                "bgcolor": "#FAFAFA",
                "bordercolor": "#E2E8E4",
                "borderwidth": 1,
            },
        ))
        fig_gauge.update_layout(
            **CHART_THEME,
            height=240,
        )
        st.plotly_chart(fig_gauge, width='stretch')

    with col_t:
        st.markdown('<p class="sec-head">Parameter Check</p>', unsafe_allow_html=True)
        st.caption("Approximate WHO/BIS guideline limits — for reference only.")

        check_results = check_parameters(input_values)
        rows_data = []
        for r in check_results:
            if r["who_min"] is not None and r["who_max"] is not None:
                guide = f"{r['who_min']} – {r['who_max']}"
            elif r["who_max"] is not None:
                guide = f"≤ {r['who_max']}"
            elif r["who_min"] is not None:
                guide = f"≥ {r['who_min']}"
            else:
                guide = "—"
            icon = {"OK": "OK", "Too High": "High", "Too Low": "Low"}[r["status"]]
            rows_data.append({
                "Parameter": r["display_name"],
                "Value":     f"{r['value']:.2f} {r['unit']}",
                "Guideline": guide,
                "Status":    icon,
            })
        st.dataframe(
            pd.DataFrame(rows_data),
            width='stretch',
            hide_index=True,
            height=230,
        )

    # ── Advice ──
    st.markdown('<p class="sec-head">Remediation Advice</p>', unsafe_allow_html=True)
    with st.spinner("Preparing advice..."):
        ai_text = get_ai_advice(input_values, prediction, safe_prob)

    if ai_text:
        st.markdown(
            f"""
            <div class="card">
                <p style="font-size:0.72rem;font-weight:700;letter-spacing:0.8px;
                   text-transform:uppercase;color:#2A7C6F;margin-bottom:0.5rem;">
                    AI Explanation — Gemini
                </p>
                <div style="font-size:0.9rem;color:#1A1A1A;">{ai_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        rule_text = build_rule_based_summary(check_results)
        st.markdown(
            f'<div class="card" style="font-size:0.9rem;white-space:pre-line;">{rule_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown(
        """
        <div style="padding:1.4rem 0 0.4rem;">
            <h2 style="font-family:'DM Serif Display',serif;font-size:1.5rem;
               color:#1C2B2A;margin:0 0 0.3rem;">Testing Dashboard</h2>
            <p style="font-size:0.88rem;color:#718096;margin:0;">
                Historical test records, safety trends, and parameter monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_history()
    if df.empty:
        st.info("No tests yet. Go to **Test Water** to run your first sample.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Result"]    = df["prediction"].map({1: "Safe", 0: "Unsafe"})

    # ── Danger alert ──
    recent = df.head(3)
    if len(recent) >= 3 and (recent["prediction"] == 0).all():
        st.markdown(
            """
            <div class="alert-danger">
                <b>Danger Alert</b> — The last 3 consecutive tests were all <b>Unsafe</b>.
                Do not use this water source. Contact your local water authority.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Metrics ──
    st.markdown('<p class="sec-head">Summary</p>', unsafe_allow_html=True)
    total    = len(df)
    n_safe   = int((df["prediction"] == 1).sum())
    n_unsafe = int((df["prediction"] == 0).sum())
    avg_prob = df["probability"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tests",   total)
    m2.metric("Safe",   n_safe,   delta=f"{n_safe/total:.0%}")
    m3.metric("Unsafe", n_unsafe, delta=f"-{n_unsafe/total:.0%}", delta_color="inverse")
    m4.metric("Avg Safe Prob", f"{avg_prob:.1%}")

    # ── Charts row ──
    st.markdown('<p class="sec-head">Distribution &amp; Trend</p>', unsafe_allow_html=True)
    col_pie, col_line = st.columns([1, 2])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Safe", "Unsafe"],
            values=[n_safe, n_unsafe],
            hole=0.52,
            marker_colors=["#1E7B4B", "#C0392B"],
            textfont_size=13,
        ))
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(
            **CHART_THEME,
            title={"text": "Safe vs Unsafe", "font": {"size": 13}, "y": 0.97},
            showlegend=False,
            height=280,
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_line:
        df_sorted = df.sort_values("timestamp")
        fig_line  = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_sorted["timestamp"],
            y=df_sorted["probability"],
            mode="lines+markers",
            name="Safe Prob",
            line={"color": "#2A7C6F", "width": 2},
            marker={
                "color": df_sorted["prediction"].map({1: "#1E7B4B", 0: "#C0392B"}),
                "size": 9,
                "line": {"color": "#FFFFFF", "width": 1.5},
            },
            fill="tozeroy",
            fillcolor="rgba(42,124,111,0.07)",
        ))
        fig_line.add_hline(
            y=0.5, line_dash="dot", line_color="#D97706",
            annotation_text="Threshold 50%",
            annotation_font={"size": 11, "color": "#D97706"},
        )
        fig_line.update_layout(
            **CHART_THEME,
            title={"text": "Safety Probability Over Time", "font": {"size": 13}},
            xaxis={**GRID, "title": ""},
            yaxis={**GRID, "range": [0, 1], "title": "P(Safe)", "tickformat": ".0%"},
            height=280,
        )
        st.plotly_chart(fig_line, width='stretch')

    # ── Parameter trends ──
    st.markdown('<p class="sec-head">Key Parameter Trends</p>', unsafe_allow_html=True)

    key_params   = ["ph", "Turbidity", "Chloramines", "Organic_carbon"]
    param_labels = {p: PARAM_CONFIG[p]["display_name"] for p in key_params}
    clr_map      = {"ph": "#2A7C6F", "Turbidity": "#D97706",
                    "Chloramines": "#C0392B", "Organic_carbon": "#1E7B4B"}

    fig_tr = go.Figure()
    for p in key_params:
        if p in df_sorted.columns:
            fig_tr.add_trace(go.Scatter(
                x=df_sorted["timestamp"],
                y=df_sorted[p],
                name=param_labels[p],
                mode="lines+markers",
                line={"color": clr_map[p], "width": 1.8},
                marker={"size": 6},
            ))
    fig_tr.update_layout(
        **CHART_THEME,
        xaxis={**GRID},
        yaxis={**GRID},
        legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "#E2E8E4",
                "borderwidth": 1, "font": {"size": 11}},
        height=320,
    )
    st.plotly_chart(fig_tr, width='stretch')

    # ── History table ──
    st.markdown('<p class="sec-head">Full History</p>', unsafe_allow_html=True)
    display_df = df[["timestamp", "Result", "probability"] + FEATURE_COLUMNS].copy()
    display_df["probability"] = display_df["probability"].map("{:.1%}".format)
    display_df = display_df.rename(columns={"timestamp": "Timestamp", "probability": "Safe Prob"})
    st.dataframe(display_df, width='stretch', hide_index=True, height=320)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv_bytes,
                       file_name="aquasafe_history.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
def page_model_info():
    st.markdown(
        """
        <div style="padding:1.4rem 0 0.4rem;">
            <h2 style="font-family:'DM Serif Display',serif;font-size:1.5rem;
               color:#1C2B2A;margin:0 0 0.3rem;">Model Information</h2>
            <p style="font-size:0.88rem;color:#718096;margin:0;">
                Performance metrics, confusion matrix and feature importances of the trained classifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metrics ──
    mp = pathlib.Path(METRICS_PATH)
    if mp.exists():
        st.markdown('<p class="sec-head">Test-Set Metrics</p>', unsafe_allow_html=True)
        df_m       = pd.read_csv(mp)
        model_name = df_m["Model"].iloc[0] if "Model" in df_m.columns else "Best Model"
        st.markdown(
            f"""
            <div class="card">
                <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.8px;
                   text-transform:uppercase;color:#718096;">Best Model</span>
                <p style="font-size:1.1rem;font-weight:700;color:#1C2B2A;margin:4px 0 0;">
                    {model_name}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = [c for c in df_m.columns if c != "Model"]
        mcols = st.columns(len(metric_cols))
        for col, mc in zip(mcols, metric_cols):
            label = "Unsafe Recall" if mc == "Unsafe_Recall" else mc
            col.metric(label, f"{df_m[mc].iloc[0]:.4f}")
    else:
        st.warning("Metrics file not found. Run `python train.py` first.")

    # ── Images ──
    st.markdown('<p class="sec-head">Visualisations</p>', unsafe_allow_html=True)
    c_cm, c_fi = st.columns(2)
    with c_cm:
        cm_path = pathlib.Path(CM_IMAGE_PATH)
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", width='stretch')
        else:
            st.info("Run `python train.py` to generate the confusion matrix.")
    with c_fi:
        fi_path = pathlib.Path(FI_IMAGE_PATH)
        if fi_path.exists():
            st.image(str(fi_path), caption="Feature Importance", width='stretch')
        else:
            st.info("Run `python train.py` to generate feature importances.")

    # ── About ──
    st.markdown('<p class="sec-head">About the Model</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p class="card-title">Pipeline Architecture</p>
            <ol style="color:#4A5568;line-height:1.9;font-size:0.88rem;margin:0.5rem 0 0;">
                <li><b>Median Imputation</b> — fills missing values with per-feature medians</li>
                <li><b>Standard Scaling</b> — zero-mean, unit-variance normalisation</li>
                <li><b>Classifier</b> — best of Logistic Regression, Random Forest, XGBoost 
                    (selected by cross-validated ROC-AUC)</li>
            </ol>
        </div>
        <div class="card">
            <p class="card-title">Class Imbalance Handling</p>
            <p style="color:#4A5568;font-size:0.88rem;margin:0.4rem 0 0;">
                The dataset has roughly 61% unsafe / 39% safe samples. This is addressed via
                <code>class_weight="balanced"</code> and optimising for ROC-AUC rather than accuracy.
                We also track <b>Recall for the Unsafe class</b> separately — a false negative 
                (calling unsafe water safe) is a public health risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_header()
    page = render_nav()

    if page == "Test Water":
        page_test_water()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Model Info":
        page_model_info()


if __name__ == "__main__":
    main()
