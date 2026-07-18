import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from textwrap import dedent
import base64

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
SHAP_DIR = OUTPUT_DIR / "shap"

# ============================================================
# PALETTE
# ============================================================
C = {
    "bg": "#08090B",
    "bg2": "#101216",
    "card": "#15171B",
    "card_elevated": "#1B1E23",
    "accent": "#F2A93B",
    "gold": "#E7C77A",
    "copper": "#C77A45",
    "critical": "#E05252",
    "coral": "#F0785A",
    "text": "#F5F1E8",
    "text2": "#A8A9AD",
    "muted": "#74777D",
    "low": "#3BA87A",
    "warning": "#F2A93B",
    "high": "#E05252",
    "tda": "#F2A93B",
    "market": "#5B8DEF",
    "grid": "rgba(116,119,125,0.08)",
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TDA Crash Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS
# ============================================================
def inject_css():
    st.markdown(
        dedent("""\
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

        :root {
            --bg: #08090B;
            --bg2: #101216;
            --card: #15171B;
            --card-elev: #1B1E23;
            --accent: #F2A93B;
            --gold: #E7C77A;
            --copper: #C77A45;
            --critical: #E05252;
            --coral: #F0785A;
            --text: #F5F1E8;
            --text2: #A8A9AD;
            --muted: #74777D;
            --low: #3BA87A;
        }

        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"], .main, .block-container {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'DM Sans', 'Inter', sans-serif !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D0E12 0%, #101216 100%) !important;
            border-right: 1px solid rgba(242,169,59,0.1) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            color: var(--text) !important;
        }

        .block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #15171B 0%, #1B1E23 100%);
            border: 1px solid rgba(242,169,59,0.12);
            border-radius: 12px;
            padding: 20px 22px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
            box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        }
        .metric-card:hover {
            border-color: rgba(242,169,59,0.35);
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(242,169,59,0.08);
        }
        .metric-card .metric-label {
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #A8A9AD;
            margin-bottom: 6px;
        }
        .metric-card .metric-value {
            font-family: 'Space Grotesk', monospace;
            font-size: 28px;
            font-weight: 700;
            color: #F2A93B;
            line-height: 1.1;
        }
        .metric-card .metric-sub {
            font-size: 11px;
            color: #74777D;
            margin-top: 4px;
        }

        /* Section headers */
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #F2A93B;
            border-bottom: 1px solid rgba(242,169,59,0.15);
            padding-bottom: 8px;
            margin-bottom: 16px;
            margin-top: 24px;
        }

        /* Risk badges */
        .risk-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .risk-low { background: rgba(59,168,122,0.15); color: #3BA87A; border: 1px solid rgba(59,168,122,0.3); }
        .risk-warning { background: rgba(242,169,59,0.15); color: #F2A93B; border: 1px solid rgba(242,169,59,0.3); }
        .risk-high {
            background: rgba(224,82,82,0.15); color: #E05252; border: 1px solid rgba(224,82,82,0.3);
            animation: risk-pulse 2s ease-in-out infinite;
        }

        @keyframes risk-pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(224,82,82,0); }
            50% { box-shadow: 0 0 12px 2px rgba(224,82,82,0.25); }
        }

        /* Info cards */
        .info-card {
            background: linear-gradient(135deg, #15171B 0%, #1B1E23 100%);
            border: 1px solid rgba(242,169,59,0.10);
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 12px;
        }
        .info-card h4 { margin-top: 0; font-size: 15px; color: #E7C77A; }
        .info-card p { font-size: 13.5px; line-height: 1.65; color: #A8A9AD; margin-bottom: 0; }

        /* Pipeline step */
        .pipe-step {
            background: #15171B;
            border: 1px solid rgba(242,169,59,0.10);
            border-radius: 10px;
            padding: 12px 16px;
            text-align: center;
            font-size: 12.5px;
            font-weight: 500;
            color: #A8A9AD;
            margin-bottom: 6px;
        }
        .pipe-arrow { text-align: center; color: #F2A93B; font-size: 18px; margin: 2px 0; }

        /* Streamlit overrides */
        .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid rgba(242,169,59,0.12); }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #A8A9AD !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 13px !important;
            font-weight: 500;
            letter-spacing: 0.5px;
            padding: 10px 20px !important;
            border-bottom: 2px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #F2A93B !important;
            border-bottom: 2px solid #F2A93B !important;
        }

        [data-testid="stExpander"] {
            background: #15171B !important;
            border: 1px solid rgba(242,169,59,0.08) !important;
            border-radius: 10px !important;
        }

        .stDownloadButton button, .stButton button {
            background: linear-gradient(135deg, #F2A93B 0%, #C77A45 100%) !important;
            color: #08090B !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
            font-size: 13px !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(242,169,59,0.25) !important;
        }

        .stSelectbox > div > div,
        .stDateInput > div > div {
            background: #15171B !important;
            border: 1px solid rgba(242,169,59,0.15) !important;
            border-radius: 8px !important;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #15171B 0%, #1B1E23 100%);
            border: 1px solid rgba(242,169,59,0.12);
            border-radius: 12px;
            padding: 14px 18px;
        }
        div[data-testid="stMetric"] label { color: #A8A9AD !important; font-size: 12px !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #F2A93B !important;
            font-family: 'Space Grotesk', monospace !important;
        }

        /* Sidebar brand */
        .sidebar-brand {
            text-align: center;
            padding: 18px 12px 14px 12px;
            border-bottom: 1px solid rgba(242,169,59,0.12);
            margin-bottom: 18px;
        }
        .sidebar-brand .brand-icon { font-size: 28px; margin-bottom: 4px; }
        .sidebar-brand .brand-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 16px;
            font-weight: 700;
            color: #F2A93B;
            letter-spacing: 2px;
        }
        .sidebar-brand .brand-sub {
            font-size: 10.5px;
            color: #74777D;
            line-height: 1.5;
            margin-top: 4px;
        }

        /* Nav items */
        .nav-item {
            display: block;
            padding: 9px 16px;
            margin: 2px 8px;
            border-radius: 8px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13.5px;
            font-weight: 500;
            color: #A8A9AD;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .nav-item:hover { background: rgba(242,169,59,0.06); color: #F5F1E8; }
        .nav-active {
            background: rgba(242,169,59,0.10) !important;
            color: #F2A93B !important;
            border-left: 3px solid #F2A93B;
        }

        /* Hero signal card */
        .hero-signal {
            background: linear-gradient(135deg, #15171B 0%, #1B1E23 100%);
            border: 1px solid rgba(242,169,59,0.15);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
            position: relative;
            overflow: hidden;
        }
        .hero-signal::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(242,169,59,0.03) 0%, transparent 60%);
            pointer-events: none;
        }

        /* Feature tag */
        .feature-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            margin: 2px 3px;
            letter-spacing: 0.3px;
        }
        .tag-tda { background: rgba(242,169,59,0.12); color: #F2A93B; border: 1px solid rgba(242,169,59,0.2); }
        .tag-market { background: rgba(91,141,239,0.12); color: #5B8DEF; border: 1px solid rgba(91,141,239,0.2); }

        /* Gauge container */
        .gauge-container {
            text-align: center;
            padding: 10px 0;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #08090B; }
        ::-webkit-scrollbar-thumb { background: #2A2D35; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #F2A93B; }

        /* Fade-in animation */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeUp 0.5s ease-out; }

        /* Hide Streamlit defaults */
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden !important; }

        .sidebar-arch {
            background: #15171B;
            border: 1px solid rgba(242,169,59,0.10);
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
            font-size: 11.5px;
        }
        .sidebar-arch .arch-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #74777D;
            margin-bottom: 8px;
        }
        .sidebar-arch .arch-step {
            text-align: center;
            color: #A8A9AD;
            font-size: 11.5px;
            padding: 3px 0;
        }
        .sidebar-arch .arch-arrow {
            text-align: center;
            color: #F2A93B;
            font-size: 12px;
        }
        </style>
        """),
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(filename, required=False):
    path = DATA_DIR / filename
    if not path.exists():
        if required:
            st.error(f"Required project artifact could not be loaded: **data/{filename}**. "
                     f"Please ensure this file exists in the data/ directory.")
            st.stop()
        return None
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception as e:
        if required:
            st.error(f"Error reading data/{filename}: {e}")
            st.stop()
        return None


@st.cache_data(show_spinner=False)
def load_all_data():
    data = {}
    data["predictions"] = load_csv("final_test_predictions.csv", required=True)
    data["backtest_daily"] = load_csv("historical_backtest_daily.csv", required=True)
    data["metrics"] = load_csv("final_metrics.csv", required=True)
    data["shap_importance"] = load_csv("shap_feature_importance.csv", required=True)
    data["tda_shap"] = load_csv("tda_shap_importance.csv")
    data["highest_risk_shap"] = load_csv("highest_risk_shap_explanation.csv")
    data["crash_episodes"] = load_csv("crash_episode_backtest.csv")
    data["ablation"] = load_csv("ablation_results.csv")
    data["threshold_analysis"] = load_csv("final_threshold_analysis.csv")
    data["model_features"] = load_csv("final_model_features.csv")
    data["market_data"] = load_csv("market_data_final_features.csv")
    return data


def check_image(path):
    return Path(path).exists()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_risk(prob):
    if prob >= 0.62:
        return "HIGH", C["high"]
    elif prob >= 0.37:
        return "WARNING", C["warning"]
    else:
        return "LOW", C["low"]


def risk_badge_html(level):
    cls_map = {"LOW": "risk-low", "WARNING": "risk-warning", "HIGH": "risk-high"}
    cls = cls_map.get(level, "risk-low")
    return f'<span class="risk-badge {cls}">{level}</span>'


def metric_card(label, value, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_card(title, body):
    st.markdown(
        f'<div class="info-card"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def plotly_layout(fig, title="", height=420, showlegend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=15, color=C["text"]), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text2"], size=11),
        height=height,
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C["text2"])),
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor=C["grid"], zerolinecolor=C["grid"],
            tickfont=dict(size=10, color=C["muted"]),
        ),
        yaxis=dict(
            gridcolor=C["grid"], zerolinecolor=C["grid"],
            tickfont=dict(size=10, color=C["muted"]),
        ),
        hoverlabel=dict(
            bgcolor=C["card_elevated"],
            font_size=12,
            font_family="DM Sans",
            font_color=C["text"],
            bordercolor=C["accent"],
        ),
    )
    return fig


def is_tda_feature(name):
    return str(name).startswith("TDA_")


def pct_fmt(val):
    return f"{val * 100:.2f}%"


# ============================================================
# GAUGE CHART
# ============================================================
def create_gauge(prob, size=280):
    level, color = classify_risk(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=36, color=color, family="Space Grotesk")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=C["muted"],
                      tickfont=dict(size=9, color=C["muted"])),
            bar=dict(color=color, thickness=0.3),
            bgcolor=C["card"],
            borderwidth=0,
            steps=[
                dict(range=[0, 37], color="rgba(59,168,122,0.08)"),
                dict(range=[37, 62], color="rgba(242,169,59,0.08)"),
                dict(range=[62, 100], color="rgba(224,82,82,0.08)"),
            ],
            threshold=dict(line=dict(color=C["accent"], width=2), thickness=0.8, value=prob * 100),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text2"]),
        height=size,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
PAGES = [
    ("◆ Overview", "overview"),
    ("◈ Market Analysis", "market"),
    ("◇ TDA Intelligence", "tda"),
    ("▣ Crash Prediction", "prediction"),
    ("▤ Explainability", "explainability"),
    ("▥ Backtesting", "backtest"),
    ("▦ Model Performance", "performance"),
    ("▧ Methodology", "methodology"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="brand-icon">◆</div>'
            '<div class="brand-title">TDA CRASH INTELLIGENCE</div>'
            '<div class="brand-sub">Topological early-warning system for<br>structural market instability</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        for label, key in PAGES:
            is_active = st.session_state.page == key
            if st.sidebar.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        st.markdown(
            '<div class="sidebar-arch">'
            '<div class="arch-title">Final Architecture</div>'
            '<div class="arch-step">Market Features</div>'
            '<div class="arch-arrow">↓</div>'
            '<div class="arch-step">Persistent Homology</div>'
            '<div class="arch-arrow">↓</div>'
            '<div class="arch-step">CatBoost</div>'
            '<div class="arch-arrow">↓</div>'
            '<div class="arch-step" style="color:#F2A93B;">Crash Risk Probability</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:10.5px; color:#74777D; text-align:center;">'
            '<span style="color:#F2A93B;">Warning</span> ≥ 0.37&emsp;'
            '<span style="color:#E05252;">High Risk</span> ≥ 0.62'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: OVERVIEW
# ============================================================
def page_overview(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:26px; letter-spacing:3px; margin-bottom:2px;">TOPOLOGICAL MARKET CRASH INTELLIGENCE</h1>'
        '<p style="color:#A8A9AD; font-size:13.5px; max-width:900px; line-height:1.7;">'
        'A machine-learning early-warning framework combining market dynamics with persistent homology '
        'to identify structural instability associated with severe future market declines.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Latest signal
    pred = data["predictions"]
    latest = pred.iloc[-1]
    latest_date = pd.to_datetime(latest["Date"]).strftime("%Y-%m-%d") if pd.notna(latest["Date"]) else "N/A"
    latest_prob = float(latest["Crash_Probability"])
    level, color = classify_risk(latest_prob)

    section_header("Latest Available Out-of-Sample Signal")

    col_gauge, col_info = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(create_gauge(latest_prob, size=260), width="stretch", config={"displayModeBar": False})
    with col_info:
        st.markdown(
            f'<div class="hero-signal">'
            f'<div style="font-size:11px; color:{C["muted"]}; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;">'
            f'Observation Date</div>'
            f'<div style="font-size:22px; font-family:Space Grotesk; font-weight:600; color:{C["text"]}; margin-bottom:14px;">'
            f'{latest_date}</div>'
            f'<div style="font-size:11px; color:{C["muted"]}; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">'
            f'Crash Probability</div>'
            f'<div style="font-size:36px; font-family:Space Grotesk; font-weight:700; color:{color}; margin-bottom:14px;">'
            f'{latest_prob:.4f}</div>'
            f'<div>{risk_badge_html(level)}</div>'
            f'<div style="font-size:11px; color:{C["muted"]}; margin-top:10px;">'
            f'Historical out-of-sample prediction &mdash; not a live market signal.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Key metrics
    section_header("Key Performance Metrics")
    mc = st.columns(5)
    metrics_map = [
        ("ROC-AUC", "0.8365", "Discrimination"),
        ("PR-AUC", "0.1547", "Rare-event detection"),
        ("TDA Contribution", "38.12%", "SHAP attribution"),
        ("Final Features", "29", "Market + TDA"),
        ("Best F1 Threshold", "0.62", "Optimized cutoff"),
    ]
    for i, (label, val, sub) in enumerate(metrics_map):
        with mc[i]:
            metric_card(label, val, sub)

    # Crash probability timeline
    section_header("Crash Probability Timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["Date"], y=pred["Crash_Probability"],
        mode="lines", line=dict(color=C["accent"], width=1.2),
        name="Crash Probability", fill="tozeroy",
        fillcolor="rgba(242,169,59,0.06)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Probability: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.37, line_dash="dot", line_color=C["warning"], line_width=1,
                  annotation_text="Warning (0.37)", annotation_font_color=C["warning"],
                  annotation_font_size=10, annotation_position="top left")
    fig.add_hline(y=0.62, line_dash="dot", line_color=C["critical"], line_width=1,
                  annotation_text="High Risk (0.62)", annotation_font_color=C["critical"],
                  annotation_font_size=10, annotation_position="top left")
    plotly_layout(fig, "", height=340, showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Risk distribution & recent high-risk
    c1, c2 = st.columns(2)
    with c1:
        section_header("Prediction Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pred["Crash_Probability"], nbinsx=60,
            marker_color=C["accent"], opacity=0.75,
            hovertemplate="Probability: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=0.37, line_dash="dot", line_color=C["warning"], line_width=1)
        fig_hist.add_vline(x=0.62, line_dash="dot", line_color=C["critical"], line_width=1)
        plotly_layout(fig_hist, "", height=300, showlegend=False)
        fig_hist.update_layout(xaxis_title="Crash Probability", yaxis_title="Count")
        st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

    with c2:
        section_header("Recent High-Risk Observations")
        high_risk = pred[pred["Crash_Probability"] >= 0.62].copy()
        if len(high_risk) > 0:
            high_risk = high_risk.sort_values("Crash_Probability", ascending=False).head(10)
            display_df = high_risk[["Date", "Crash_Probability", "Actual"]].copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["Date", "Probability", "Actual Crash"]
            st.dataframe(display_df, width="stretch", hide_index=True, height=300)
        else:
            warning_obs = pred[pred["Crash_Probability"] >= 0.37].copy()
            if len(warning_obs) > 0:
                warning_obs = warning_obs.sort_values("Crash_Probability", ascending=False).head(10)
                display_df = warning_obs[["Date", "Crash_Probability", "Actual"]].copy()
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                display_df.columns = ["Date", "Probability", "Actual Crash"]
                st.dataframe(display_df, width="stretch", hide_index=True, height=300)
            else:
                st.info("No observations above warning threshold in saved predictions.")

    # Short model summary
    section_header("Model Summary")
    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Architecture",
            "CatBoost gradient boosting classifier trained on 29 features combining traditional market "
            "indicators (returns, volatility, momentum) with 9 topological features extracted via persistent "
            "homology from 30-day rolling market windows."
        )
    with c2:
        info_card(
            "Key Scientific Finding",
            "TDA_H0_TotalPersistence is the single strongest global SHAP feature, demonstrating that "
            "topological structure of multi-asset market data captures crash-predictive information "
            "beyond traditional indicators. TDA features collectively contribute 38.12% of total SHAP importance."
        )


# ============================================================
# PAGE: MARKET ANALYSIS
# ============================================================
def page_market(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">MARKET ANALYSIS</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">Interactive exploration of underlying market data used in crash prediction.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    mkt = data.get("market_data")
    if mkt is None:
        st.warning("Market data file not available.")
        return

    # Date filter
    section_header("Date Range Selection")
    min_d = mkt["Date"].min().date()
    max_d = mkt["Date"].max().date()
    col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
    with col_d1:
        start_date = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d, key="mkt_start")
    with col_d2:
        end_date = st.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d, key="mkt_end")
    with col_d3:
        st.markdown("")

    mask = (mkt["Date"].dt.date >= start_date) & (mkt["Date"].dt.date <= end_date)
    filtered = mkt[mask].copy()

    if len(filtered) == 0:
        st.warning("No data in selected range.")
        return

    tabs = st.tabs(["Price Timelines", "Volatility & Momentum", "Cross-Market Correlation"])

    # Tab 1: Price timelines
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            if "SP500" in filtered.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered["Date"], y=filtered["SP500"],
                    mode="lines", line=dict(color=C["accent"], width=1.3),
                    name="S&P 500",
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>S&P 500: %{y:,.2f}<extra></extra>",
                ))
                plotly_layout(fig, "S&P 500", height=360, showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with c2:
            if "VIX" in filtered.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered["Date"], y=filtered["VIX"],
                    mode="lines", line=dict(color=C["coral"], width=1.3),
                    name="VIX",
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>VIX: %{y:.2f}<extra></extra>",
                ))
                plotly_layout(fig, "VIX (Volatility Index)", height=360, showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            if "NASDAQ" in filtered.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered["Date"], y=filtered["NASDAQ"],
                    mode="lines", line=dict(color=C["gold"], width=1.3),
                    name="NASDAQ",
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>NASDAQ: %{y:,.2f}<extra></extra>",
                ))
                plotly_layout(fig, "NASDAQ Composite", height=360, showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with c4:
            cols_extra = []
            if "GOLD" in filtered.columns:
                cols_extra.append(("GOLD", C["gold"]))
            if "OIL" in filtered.columns:
                cols_extra.append(("OIL", C["copper"]))
            if cols_extra:
                fig = go.Figure()
                for col_name, col_color in cols_extra:
                    fig.add_trace(go.Scatter(
                        x=filtered["Date"], y=filtered[col_name],
                        mode="lines", line=dict(color=col_color, width=1.3),
                        name=col_name,
                        hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{col_name}: %{{y:,.2f}}<extra></extra>",
                    ))
                plotly_layout(fig, "Gold & Oil", height=360)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # SP500 vs VIX dual axis
        if "SP500" in filtered.columns and "VIX" in filtered.columns:
            section_header("S&P 500 vs VIX Comparison")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=filtered["Date"], y=filtered["SP500"],
                mode="lines", line=dict(color=C["accent"], width=1.2),
                name="S&P 500",
                hovertemplate="S&P 500: %{y:,.2f}<extra></extra>",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=filtered["Date"], y=filtered["VIX"],
                mode="lines", line=dict(color=C["coral"], width=1.2),
                name="VIX",
                hovertemplate="VIX: %{y:.2f}<extra></extra>",
            ), secondary_y=True)
            plotly_layout(fig, "", height=380)
            fig.update_yaxes(title_text="S&P 500", secondary_y=False, gridcolor=C["grid"],
                             tickfont=dict(color=C["muted"], size=10))
            fig.update_yaxes(title_text="VIX", secondary_y=True, gridcolor=C["grid"],
                             tickfont=dict(color=C["muted"], size=10))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Tab 2: Volatility & Momentum
    with tabs[1]:
        vol_cols = [c for c in ["SP500_Volatility_10", "SP500_Volatility_20", "NASDAQ_Volatility_20"] if c in filtered.columns]
        if vol_cols:
            section_header("Rolling Volatility")
            fig = go.Figure()
            colors_v = [C["accent"], C["gold"], C["copper"]]
            for i, vc in enumerate(vol_cols):
                fig.add_trace(go.Scatter(
                    x=filtered["Date"], y=filtered[vc],
                    mode="lines", line=dict(color=colors_v[i % 3], width=1.2),
                    name=vc.replace("_", " "),
                    hovertemplate=f"{vc}: %{{y:.4f}}<extra></extra>",
                ))
            plotly_layout(fig, "", height=360)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        mom_cols = [c for c in ["SP500_Momentum_5", "SP500_Momentum_10", "SP500_Momentum_20"] if c in filtered.columns]
        if mom_cols:
            section_header("Momentum Indicators")
            fig = go.Figure()
            colors_m = [C["accent"], C["copper"], C["coral"]]
            for i, mc_col in enumerate(mom_cols):
                fig.add_trace(go.Scatter(
                    x=filtered["Date"], y=filtered[mc_col],
                    mode="lines", line=dict(color=colors_m[i % 3], width=1.2),
                    name=mc_col.replace("_", " "),
                    hovertemplate=f"{mc_col}: %{{y:.4f}}<extra></extra>",
                ))
            fig.add_hline(y=0, line_dash="dash", line_color=C["muted"], line_width=0.8)
            plotly_layout(fig, "", height=360)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Tab 3: Correlation
    with tabs[2]:
        section_header("Cross-Market Returns Correlation")
        return_cols = [c for c in ["SP500_Return", "NASDAQ_Return", "GOLD_Return", "OIL_Return", "VIX_Change"]
                       if c in filtered.columns]
        if return_cols:
            corr = filtered[return_cols].corr()
            labels = [c.replace("_Return", "").replace("_Change", " Chg") for c in return_cols]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale=[[0, "#1B1E23"], [0.5, C["copper"]], [1, C["accent"]]],
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                textfont=dict(size=11, color=C["text"]),
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
            ))
            plotly_layout(fig, "", height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Return columns not available for correlation analysis.")


# ============================================================
# PAGE: TDA INTELLIGENCE
# ============================================================
def page_tda(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">TDA INTELLIGENCE</h1>'
        '<p style="color:#A8A9AD; font-size:13px; max-width:900px; line-height:1.7;">'
        'Topological Data Analysis captures the <em>shape</em> and <em>structure</em> of multidimensional '
        'market behavior &mdash; information invisible to traditional statistical indicators.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Conceptual explanation
    section_header("Why Topology?")
    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Traditional Indicators",
            "Measure individual market statistics: returns, moving averages, volatility. "
            "These capture point-wise or pairwise relationships but miss higher-order structural patterns "
            "in the joint behaviour of multiple assets."
        )
    with c2:
        info_card(
            "Topological Data Analysis",
            "Analyzes the SHAPE and STRUCTURE of multi-asset market data using persistent homology. "
            "A 30-day rolling window of market returns is treated as a point cloud. The Vietoris-Rips complex "
            "reveals connected components (H0) and loops (H1) that characterize market regimes."
        )

    # H0 / H1 explanation
    section_header("Homology Groups")
    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "H0 — Connected Components",
            "Measures the clustering structure of market data points. High H0 persistence "
            "indicates distinct market regimes or fragmented behavior. "
            "TDA_H0_TotalPersistence is the single strongest global SHAP feature in the final model."
        )
    with c2:
        info_card(
            "H1 — Loops / Cycles",
            "Captures cyclic or periodic structures in multi-asset dynamics. "
            "H1 features detect recurring patterns and feedback loops in market behavior "
            "that may precede instability. H1 entropy measures the complexity of these cyclic structures."
        )

    # TDA Features table
    section_header("TDA Feature Set (9 Features)")
    tda_features = [
        ("TDA_H0_Count", "H0", "Number of connected components"),
        ("TDA_H0_TotalPersistence", "H0", "Sum of all H0 lifetimes — strongest global SHAP feature"),
        ("TDA_H0_MaxPersistence", "H0", "Maximum H0 feature lifetime"),
        ("TDA_H0_MeanPersistence", "H0", "Average H0 feature lifetime"),
        ("TDA_H1_Count", "H1", "Number of loops detected"),
        ("TDA_H1_TotalPersistence", "H1", "Sum of all H1 lifetimes"),
        ("TDA_H1_MaxPersistence", "H1", "Maximum H1 feature lifetime"),
        ("TDA_H1_MeanPersistence", "H1", "Average H1 feature lifetime"),
        ("TDA_H1_Entropy", "H1", "Shannon entropy of H1 persistence — measures cyclic complexity"),
    ]
    tda_df = pd.DataFrame(tda_features, columns=["Feature", "Group", "Description"])
    st.dataframe(tda_df, width="stretch", hide_index=True)

    # TDA contribution metric
    section_header("TDA Contribution to Model Predictions")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Overall TDA SHAP Contribution", "38.12%", "Of total model importance")
    with c2:
        metric_card("Strongest Feature", "TDA_H0_TotalPersistence", "#1 global SHAP rank")
    with c3:
        metric_card("TDA Features", "9", "Out of 29 total features")

    # TDA SHAP chart
    tda_shap = data.get("tda_shap")
    if tda_shap is not None and len(tda_shap) > 0:
        section_header("TDA Feature SHAP Importance")
        tda_sorted = tda_shap.sort_values("Mean_Abs_SHAP", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tda_sorted["Feature"],
            x=tda_sorted["Mean_Abs_SHAP"],
            orientation="h",
            marker=dict(
                color=tda_sorted["Mean_Abs_SHAP"],
                colorscale=[[0, C["copper"]], [0.5, C["accent"]], [1, C["gold"]]],
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        plotly_layout(fig, "", height=380, showlegend=False)
        fig.update_layout(xaxis_title="Mean |SHAP Value|", yaxis_title="")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # TDA vs Market contribution pie
    section_header("TDA vs Traditional Feature Contribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=["TDA Features (9)", "Market Features (20)"],
            values=[38.12, 61.88],
            marker=dict(colors=[C["accent"], C["market"]]),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=12, color=C["text"]),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        plotly_layout(fig, "", height=340)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with c2:
        # Ablation comparison
        ablation = data.get("ablation")
        if ablation is not None and len(ablation) > 0:
            abl_disp = ablation[ablation["Model"].isin(["Market Only", "Market + TDA"])].copy()
            if len(abl_disp) == 2:
                fig = go.Figure()
                for i, metric in enumerate(["ROC_AUC", "PR_AUC"]):
                    if metric in abl_disp.columns:
                        fig.add_trace(go.Bar(
                            x=abl_disp["Model"],
                            y=abl_disp[metric],
                            name=metric.replace("_", "-"),
                            marker_color=[C["copper"], C["accent"]] if i == 0 else [C["muted"], C["gold"]],
                            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                        ))
                plotly_layout(fig, "Ablation: Effect of Adding TDA", height=340)
                fig.update_layout(barmode="group")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # TDA features over time
    mkt = data.get("market_data")
    if mkt is not None and "TDA_H0_TotalPersistence" in mkt.columns:
        section_header("TDA Features Over Time")
        tda_time_cols = [c for c in ["TDA_H0_TotalPersistence", "TDA_H0_MeanPersistence", "TDA_H1_Entropy"]
                         if c in mkt.columns]
        if tda_time_cols:
            fig = go.Figure()
            colors_t = [C["accent"], C["gold"], C["copper"]]
            for i, tc in enumerate(tda_time_cols):
                fig.add_trace(go.Scatter(
                    x=mkt["Date"], y=mkt[tc],
                    mode="lines", line=dict(color=colors_t[i], width=1.1),
                    name=tc.replace("TDA_", ""),
                    hovertemplate=f"{tc}: %{{y:.4f}}<extra></extra>",
                ))
            plotly_layout(fig, "", height=360)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Methodology pipeline
    section_header("TDA Pipeline")
    steps = [
        "30-Day Rolling Market Window",
        "Multi-Asset Point Cloud",
        "Vietoris-Rips Complex Construction",
        "Persistent Homology Computation",
        "Persistence Feature Extraction (9 features)",
        "CatBoost Input Features",
    ]
    html_steps = ""
    for i, s in enumerate(steps):
        html_steps += f'<div class="pipe-step">{s}</div>'
        if i < len(steps) - 1:
            html_steps += '<div class="pipe-arrow">↓</div>'
    st.markdown(html_steps, unsafe_allow_html=True)


# ============================================================
# PAGE: CRASH PREDICTION
# ============================================================
def page_prediction(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">CRASH PREDICTION</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">Explore historical out-of-sample crash probability predictions.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    pred = data["predictions"]
    backtest = data.get("backtest_daily")

    # Date selector
    section_header("Select Prediction Date")
    dates_list = pred["Date"].dropna().sort_values().dt.date.tolist()
    selected_date = st.selectbox(
        "Observation Date",
        options=dates_list,
        index=len(dates_list) - 1,
        key="pred_date_select",
        label_visibility="collapsed",
    )

    row = pred[pred["Date"].dt.date == selected_date]
    if len(row) == 0:
        st.warning("No prediction found for this date.")
        return

    row = row.iloc[0]
    prob = float(row["Crash_Probability"])
    level, color = classify_risk(prob)
    actual = int(row["Actual"]) if pd.notna(row.get("Actual")) else None

    # Display
    c_gauge, c_detail = st.columns([1, 2])
    with c_gauge:
        st.plotly_chart(create_gauge(prob, size=260), width="stretch", config={"displayModeBar": False})

    with c_detail:
        actual_text = ""
        if actual is not None:
            actual_label = "Yes" if actual == 1 else "No"
            actual_text = (
                f'<div style="margin-top:10px; font-size:12px; color:{C["text2"]};">'
                f'Actual Crash Risk Label: <strong style="color:{C["critical"] if actual == 1 else C["low"]};">'
                f'{actual_label}</strong></div>'
            )

        # Get SP500 and VIX from backtest if available
        sp_vix_html = ""
        if backtest is not None:
            bt_row = backtest[backtest["Date"].dt.date == selected_date]
            if len(bt_row) > 0:
                bt_row = bt_row.iloc[0]
                sp_val = bt_row.get("SP500")
                vix_val = bt_row.get("VIX")
                parts = []
                if pd.notna(sp_val):
                    parts.append(f"S&P 500: <strong>{sp_val:,.2f}</strong>")
                if pd.notna(vix_val):
                    parts.append(f"VIX: <strong>{vix_val:.2f}</strong>")
                if parts:
                    sp_vix_html = (
                        f'<div style="margin-top:8px; font-size:12px; color:{C["text2"]};">'
                        + " &emsp; ".join(parts) + "</div>"
                    )

        st.markdown(
            f'<div class="hero-signal">'
            f'<div style="font-size:11px; color:{C["muted"]}; letter-spacing:1.5px; text-transform:uppercase;">Date</div>'
            f'<div style="font-size:20px; font-family:Space Grotesk; font-weight:600; color:{C["text"]}; margin-bottom:12px;">'
            f'{selected_date}</div>'
            f'<div style="font-size:11px; color:{C["muted"]}; letter-spacing:1.5px; text-transform:uppercase;">Crash Probability</div>'
            f'<div style="font-size:32px; font-family:Space Grotesk; font-weight:700; color:{color}; margin-bottom:10px;">'
            f'{prob:.6f}</div>'
            f'{risk_badge_html(level)}'
            f'{actual_text}'
            f'{sp_vix_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Timeline with thresholds and actual crash dates
    section_header("Probability Timeline with Crash Events")
    fig = go.Figure()

    # Actual crash periods
    crash_dates = pred[pred["Actual"] == 1]
    if len(crash_dates) > 0:
        fig.add_trace(go.Scatter(
            x=crash_dates["Date"], y=crash_dates["Crash_Probability"],
            mode="markers",
            marker=dict(color=C["critical"], size=6, symbol="diamond", opacity=0.7),
            name="Actual Crash Risk",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Prob: %{y:.4f}<br>ACTUAL CRASH<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=pred["Date"], y=pred["Crash_Probability"],
        mode="lines", line=dict(color=C["accent"], width=1.2),
        name="Crash Probability", fill="tozeroy",
        fillcolor="rgba(242,169,59,0.05)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Probability: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.37, line_dash="dot", line_color=C["warning"], line_width=1,
                  annotation_text="Warning (0.37)", annotation_font_color=C["warning"],
                  annotation_font_size=10)
    fig.add_hline(y=0.62, line_dash="dot", line_color=C["critical"], line_width=1,
                  annotation_text="High Risk (0.62)", annotation_font_color=C["critical"],
                  annotation_font_size=10)
    # Highlight selected date
    fig.add_vline(x=pd.Timestamp(selected_date), line_dash="solid",
                  line_color="rgba(245,241,232,0.3)", line_width=1)
    plotly_layout(fig, "", height=380)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Threshold explanation
    section_header("Threshold Interpretation")
    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Warning Threshold — 0.37",
            "Selected as a high-recall early-warning threshold. "
            "Achieves approximately 82.93% recall with 14.17% precision. "
            "Designed to catch most genuine crash events at the cost of more false alarms."
        )
    with c2:
        info_card(
            "High Risk Threshold — 0.62",
            "Maximizes F1 in the tested threshold grid. "
            "Achieves approximately 46.34% recall, 20.65% precision, and 28.57% F1. "
            "Balances detection rate against false positive burden."
        )

    with st.expander("Why are false positives expected?"):
        st.markdown(
            f'<p style="color:{C["text2"]}; font-size:13px; line-height:1.7;">'
            "This is a <strong>rare-event detection problem</strong>. The crash rate in the test period is approximately "
            "3.46% (41 of 1,185 observations). When predicting rare events, even a good classifier will produce "
            "false positives because the base rate is very low. "
            "Accuracy alone is misleading in this context &mdash; a model that always predicts 'no crash' "
            "would achieve ~96.5% accuracy but would miss every crash. "
            "PR-AUC and recall are more informative metrics for evaluating rare-event detection.</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================
def page_explainability(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">EXPLAINABILITY</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">SHAP-based analysis of feature contributions to crash predictions.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    shap_imp = data["shap_importance"]

    # Top 20 global SHAP
    section_header("Top Global SHAP Feature Importance")
    top20 = shap_imp.head(20).sort_values("Mean_Abs_SHAP", ascending=True).copy()
    top20["is_tda"] = top20["Feature"].apply(is_tda_feature)

    colors_bar = [C["accent"] if t else C["market"] for t in top20["is_tda"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20["Feature"],
        x=top20["Mean_Abs_SHAP"],
        orientation="h",
        marker_color=colors_bar,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    plotly_layout(fig, "", height=520, showlegend=False)
    fig.update_layout(xaxis_title="Mean |SHAP Value|")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Legend
    st.markdown(
        f'<div style="font-size:12px; margin-bottom:20px;">'
        f'<span class="feature-tag tag-tda">TDA Feature</span> '
        f'<span class="feature-tag tag-market">Market Feature</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Key highlights
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TDA SHAP Contribution", "38.12%", "Combined TDA importance")
    with c2:
        metric_card("#1 Feature", "TDA_H0_TotalPersistence", "Topological feature")
    with c3:
        metric_card("#2 Feature", "SP500_Volatility_10", "Market feature")

    # Saved SHAP images
    section_header("SHAP Visualizations")
    tabs_shap = st.tabs(["SHAP Bar Plot", "SHAP Summary Plot", "Top SHAP Features"])
    img_mapping = [
        (SHAP_DIR / "shap_bar.png", "SHAP Bar Plot"),
        (SHAP_DIR / "shap_summary.png", "SHAP Summary Plot"),
        (EVAL_DIR / "top_shap_features.png", "Top SHAP Features"),
    ]
    for i, (img_path, img_title) in enumerate(img_mapping):
        with tabs_shap[i]:
            if check_image(img_path):
                st.image(str(img_path), width="stretch")
            else:
                st.info(f"{img_title} image not available at {img_path.name}")

    # Highest risk SHAP explanation
    hr_shap = data.get("highest_risk_shap")
    if hr_shap is not None and len(hr_shap) > 0:
        section_header("Highest-Risk Observation — Local SHAP Explanation")
        st.markdown(
            f'<p style="color:{C["text2"]}; font-size:13px; margin-bottom:12px;">'
            "Why did the model assign the highest crash risk probability to a specific observation?</p>",
            unsafe_allow_html=True,
        )

        hr_sorted = hr_shap.sort_values("Abs_SHAP", ascending=True).tail(15)
        hr_sorted["is_tda"] = hr_sorted["Feature"].apply(is_tda_feature)
        bar_colors = [C["critical"] if v > 0 else C["low"] for v in hr_sorted["SHAP_Value"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=hr_sorted["Feature"],
            x=hr_sorted["SHAP_Value"],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<br>Value: %{customdata:.4f}<extra></extra>",
            customdata=hr_sorted["Feature_Value"],
        ))
        plotly_layout(fig, "", height=440, showlegend=False)
        fig.update_layout(xaxis_title="SHAP Value (impact on prediction)")
        fig.add_vline(x=0, line_color=C["muted"], line_width=0.8)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f'<div style="font-size:12px; color:{C["text2"]}; line-height:1.7;">'
            f'<span style="color:{C["critical"]};">■</span> Positive SHAP &mdash; contributed to higher crash risk prediction<br>'
            f'<span style="color:{C["low"]};">■</span> Negative SHAP &mdash; reduced predicted crash risk<br><br>'
            f'<em style="color:{C["muted"]};">SHAP values indicate each feature\'s contribution to the model '
            f'prediction for this specific observation. They do not imply causality.</em></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: BACKTESTING
# ============================================================
def page_backtest(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">HISTORICAL BACKTESTING</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">Out-of-sample backtest: 2021-10-15 to 2026-07-02 (1,185 observations)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    bt = data["backtest_daily"]
    episodes = data.get("crash_episodes")

    # S&P 500 with risk signals
    section_header("S&P 500 with Risk Signals")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["Date"], y=bt["SP500"],
        mode="lines", line=dict(color=C["text2"], width=1),
        name="S&P 500",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>S&P 500: %{y:,.2f}<extra></extra>",
    ))

    # Actual crash periods
    crash_mask = bt["Actual"] == 1
    if crash_mask.any():
        fig.add_trace(go.Scatter(
            x=bt.loc[crash_mask, "Date"],
            y=bt.loc[crash_mask, "SP500"],
            mode="markers",
            marker=dict(color=C["critical"], size=5, symbol="diamond", opacity=0.6),
            name="Actual Crash Risk",
            hovertemplate="CRASH RISK<br>%{x|%Y-%m-%d}<br>S&P: %{y:,.2f}<extra></extra>",
        ))

    # Warning signals
    if "Risk_Level" in bt.columns:
        warning_mask = bt["Risk_Level"] == "WARNING"
        if warning_mask.any():
            fig.add_trace(go.Scatter(
                x=bt.loc[warning_mask, "Date"],
                y=bt.loc[warning_mask, "SP500"],
                mode="markers",
                marker=dict(color=C["warning"], size=4, symbol="triangle-up", opacity=0.5),
                name="Warning Signal",
                hovertemplate="WARNING<br>%{x|%Y-%m-%d}<br>S&P: %{y:,.2f}<extra></extra>",
            ))

        high_mask = bt["Risk_Level"] == "HIGH"
        if high_mask.any():
            fig.add_trace(go.Scatter(
                x=bt.loc[high_mask, "Date"],
                y=bt.loc[high_mask, "SP500"],
                mode="markers",
                marker=dict(color=C["critical"], size=6, symbol="x", opacity=0.8),
                name="High Risk Signal",
                hovertemplate="HIGH RISK<br>%{x|%Y-%m-%d}<br>S&P: %{y:,.2f}<extra></extra>",
            ))

    plotly_layout(fig, "", height=420)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Probability timeline backtest
    section_header("Probability Timeline — Backtest Period")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=bt["Date"], y=bt["Crash_Probability"],
        mode="lines", line=dict(color=C["accent"], width=1.1),
        name="Crash Probability", fill="tozeroy",
        fillcolor="rgba(242,169,59,0.05)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Prob: %{y:.4f}<extra></extra>",
    ))
    if crash_mask.any():
        fig2.add_trace(go.Scatter(
            x=bt.loc[crash_mask, "Date"],
            y=bt.loc[crash_mask, "Crash_Probability"],
            mode="markers",
            marker=dict(color=C["critical"], size=5, symbol="diamond", opacity=0.6),
            name="Actual Crash Risk",
        ))
    fig2.add_hline(y=0.37, line_dash="dot", line_color=C["warning"], line_width=1,
                   annotation_text="Warning (0.37)", annotation_font_color=C["warning"],
                   annotation_font_size=10)
    fig2.add_hline(y=0.62, line_dash="dot", line_color=C["critical"], line_width=1,
                   annotation_text="High Risk (0.62)", annotation_font_color=C["critical"],
                   annotation_font_size=10)
    plotly_layout(fig2, "", height=350)
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    # Crash episode table
    if episodes is not None and len(episodes) > 0:
        section_header("Crash Episode Analysis")

        st.markdown(
            f'<p style="color:{C["text2"]}; font-size:13px; line-height:1.7; margin-bottom:14px;">'
            f"The backtest identified <strong>{len(episodes)}</strong> distinct crash-risk episodes. "
            f"The table below shows whether the model provided early warnings before each episode began.</p>",
            unsafe_allow_html=True,
        )

        display_ep = episodes.copy()
        # Format for display
        date_cols = ["Episode_Start", "Episode_End", "First_Warning_Date", "First_High_Risk_Date"]
        for col in date_cols:
            if col in display_ep.columns:
                display_ep[col] = pd.to_datetime(display_ep[col], errors="coerce").dt.strftime("%Y-%m-%d")
                display_ep[col] = display_ep[col].fillna("—")

        num_cols = ["Warning_Lead_Trading_Days", "High_Risk_Lead_Trading_Days"]
        for col in num_cols:
            if col in display_ep.columns:
                display_ep[col] = display_ep[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "—"
                )

        if "Worst_10D_Drawdown" in display_ep.columns:
            display_ep["Worst_10D_Drawdown"] = display_ep["Worst_10D_Drawdown"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )
        if "Max_Probability_Before_Start" in display_ep.columns:
            display_ep["Max_Probability_Before_Start"] = display_ep["Max_Probability_Before_Start"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )

        st.dataframe(display_ep, width="stretch", hide_index=True, height=440)

        # Interpretation
        with st.expander("Interpreting Lead Times"):
            st.markdown(
                f'<p style="color:{C["text2"]}; font-size:13px; line-height:1.7;">'
                "Some lead-time values equal 30 because the analysis used a maximum 30-observation lookback window. "
                "This means the warning occurred <em>at least</em> at the tested 30-trading-day lookback boundary, "
                "not necessarily exactly 30 days before. Actual lead times may be longer.<br><br>"
                "At least one episode (2025-02-28, a single-day event) was missed entirely &mdash; no warning was "
                "generated before the crash-risk window. The model does not detect every event, and some weaknesses "
                "remain in detecting isolated single-day drawdowns with limited preceding signal.</p>",
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download Episode Data",
            episodes.to_csv(index=False),
            "crash_episode_backtest.csv",
            "text/csv",
            key="dl_episodes",
        )


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
def page_performance(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">MODEL PERFORMANCE</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">Comprehensive evaluation of the final Market + TDA CatBoost model.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Key metrics
    section_header("Performance Summary")
    mc = st.columns(6)
    perf_cards = [
        ("ROC-AUC", "0.8365", "Discrimination"),
        ("PR-AUC", "0.1547", "Rare-event"),
        ("Precision @ 0.62", "20.65%", "High-risk threshold"),
        ("Recall @ 0.62", "46.34%", "High-risk threshold"),
        ("F1 @ 0.62", "0.2857", "Balanced"),
        ("TDA Contribution", "38.12%", "SHAP-based"),
    ]
    for i, (label, val, sub) in enumerate(perf_cards):
        with mc[i]:
            metric_card(label, val, sub)

    # Context card
    st.markdown(
        f'<div class="info-card" style="margin-top:14px;">'
        f'<h4>Interpreting Metrics for Rare-Event Detection</h4>'
        f'<p>The test crash rate is approximately <strong>3.46%</strong> (41 of 1,185 observations). '
        f'For rare-event prediction, PR-AUC is more informative than ROC-AUC. '
        f'A random classifier would achieve a PR baseline of ~0.0346. '
        f'The final model\'s PR-AUC of <strong>0.1547</strong> is substantially above this baseline, '
        f'indicating meaningful predictive signal. '
        f'ROC-AUC of 0.8365 indicates good discrimination but is <strong>not</strong> accuracy.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Evaluation images
    section_header("Evaluation Curves")
    tabs_eval = st.tabs(["ROC Curve", "Precision-Recall Curve", "Confusion Matrix"])
    img_eval = [
        (EVAL_DIR / "roc_curve.png", "ROC Curve"),
        (EVAL_DIR / "precision_recall_curve.png", "Precision-Recall Curve"),
        (EVAL_DIR / "confusion_matrix.png", "Confusion Matrix"),
    ]
    for i, (img_path, img_title) in enumerate(img_eval):
        with tabs_eval[i]:
            if check_image(img_path):
                st.image(str(img_path), width="stretch")
            else:
                st.info(f"{img_title} image not found.")

    # Threshold analysis
    threshold_df = data.get("threshold_analysis")
    if threshold_df is not None and len(threshold_df) > 0:
        section_header("Threshold Analysis")
        fig = go.Figure()
        for metric, color in [("Precision", C["market"]), ("Recall", C["coral"]), ("F1", C["accent"])]:
            if metric in threshold_df.columns:
                fig.add_trace(go.Scatter(
                    x=threshold_df["Threshold"], y=threshold_df[metric],
                    mode="lines", line=dict(color=color, width=1.5),
                    name=metric,
                    hovertemplate=f"Threshold: %{{x:.2f}}<br>{metric}: %{{y:.4f}}<extra></extra>",
                ))
        fig.add_vline(x=0.37, line_dash="dot", line_color=C["warning"], line_width=1,
                      annotation_text="Warning", annotation_font_color=C["warning"],
                      annotation_font_size=10)
        fig.add_vline(x=0.62, line_dash="dot", line_color=C["critical"], line_width=1,
                      annotation_text="Best F1", annotation_font_color=C["critical"],
                      annotation_font_size=10)
        plotly_layout(fig, "", height=380)
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Ablation study
    ablation = data.get("ablation")
    if ablation is not None and len(ablation) > 0:
        section_header("Ablation Study")
        st.markdown(
            f'<p style="color:{C["text2"]}; font-size:13px; line-height:1.7; margin-bottom:14px;">'
            "Systematic comparison of feature set configurations to evaluate the contribution of each component.</p>",
            unsafe_allow_html=True,
        )

        # Grouped bar chart
        abl_metrics = ["ROC_AUC", "PR_AUC", "F1_05"]
        available_metrics = [m for m in abl_metrics if m in ablation.columns]

        fig = go.Figure()
        bar_colors = [C["muted"], C["accent"], C["copper"], C["coral"]]
        for i, _, in enumerate(ablation.iterrows()):
            row = ablation.iloc[i]
            model_name = row["Model"]
            vals = [row[m] for m in available_metrics]
            fig.add_trace(go.Bar(
                x=[m.replace("_", "-").replace("05", "@0.50") for m in available_metrics],
                y=vals,
                name=model_name,
                marker_color=bar_colors[i % len(bar_colors)],
                hovertemplate=f"{model_name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ))
        plotly_layout(fig, "", height=400)
        fig.update_layout(barmode="group", xaxis_title="Metric", yaxis_title="Score")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Ablation table
        st.dataframe(ablation, width="stretch", hide_index=True)

        # Interpretation
        with st.expander("Ablation Interpretation"):
            st.markdown(
                f'<p style="color:{C["text2"]}; font-size:13px; line-height:1.7;">'
                "<strong>Market Only → Market + TDA:</strong><br>"
                "ROC-AUC: 0.7829 → 0.8365 &emsp; PR-AUC: 0.0965 → 0.1547 &emsp; F1: 0.1783 → 0.2703<br>"
                "TDA significantly improves all key metrics, confirming topological features capture "
                "crash-predictive information beyond traditional indicators.<br><br>"
                "<strong>Market + TDA + PELT:</strong><br>"
                "Performance decreased across most metrics. PELT change-point detection did not improve "
                "the selected performance objectives.<br><br>"
                "<strong>Full Experimental Model (+ HMM):</strong><br>"
                "Improved ROC-AUC (0.8468) and recall but reduced PR-AUC (0.1301), precision, and F1 "
                "compared with Market + TDA. HMM regimes helped detect more events but at the cost of "
                "balanced rare-event performance.<br><br>"
                "<strong>Selected Architecture: Market + TDA</strong> — the strongest balanced model "
                "for rare-event prediction.</p>",
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: METHODOLOGY
# ============================================================
def page_methodology(data):
    st.markdown(
        '<div class="fade-in">'
        '<h1 style="font-size:22px; letter-spacing:2px;">METHODOLOGY</h1>'
        '<p style="color:#A8A9AD; font-size:13px;">Complete pipeline documentation and research context.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Pipeline
    section_header("Processing Pipeline")
    pipeline_steps = [
        ("1", "Market Data Collection", "S&P 500, VIX, NASDAQ, Gold, Oil"),
        ("2", "Data Cleaning", "Handle missing values, align dates, validate integrity"),
        ("3", "Feature Engineering", "Returns, log-returns, volatility (10/20-day), momentum, MA distances, VIX ratios"),
        ("4", "Crash Label Generation", "Future 10-day maximum drawdown ≥ 7% → Crash_Risk = 1"),
        ("5", "30-Day Rolling Windows", "Sliding windows of multi-asset market data as input to TDA"),
        ("6", "Persistent Homology", "Vietoris-Rips complex → H0 and H1 persistence diagrams"),
        ("7", "TDA Feature Extraction", "9 topological features: counts, persistence (total/max/mean), entropy"),
        ("8", "CatBoost Classification", "Gradient boosting on 29 combined Market + TDA features"),
        ("9", "Threshold Optimization", "Grid search for optimal warning (0.37) and high-risk (0.62) thresholds"),
        ("10", "Historical Backtesting", "Out-of-sample evaluation: 2021-10-15 to 2026-07-02"),
        ("11", "SHAP Explainability", "Global and local feature importance analysis"),
    ]

    for step_num, title, desc in pipeline_steps:
        st.markdown(
            f'<div class="pipe-step" style="text-align:left; padding:12px 18px;">'
            f'<span style="color:{C["accent"]}; font-weight:700; margin-right:10px;">{step_num}</span>'
            f'<strong style="color:{C["text"]};">{title}</strong>'
            f'<br><span style="font-size:11.5px; color:{C["muted"]};">{desc}</span></div>',
            unsafe_allow_html=True,
        )
        if step_num != "11":
            st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    # Experimental components
    section_header("Experimental Components (Not in Final Architecture)")
    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "PELT Change-Point Detection",
            "Pruned Exact Linear Time algorithm for detecting structural breaks in market data. "
            "6 additional features were extracted. Ablation showed that adding PELT reduced "
            "PR-AUC and F1, so it was not retained in the final architecture."
        )
    with c2:
        info_card(
            "HMM Market Regime Detection",
            "Hidden Markov Model for identifying latent market states. "
            "Adding HMM improved ROC-AUC and recall in the full experimental model but reduced "
            "PR-AUC and precision compared with Market + TDA. "
            "Not retained in the final selected architecture."
        )

    # Final architecture
    section_header("Final Selected Architecture")
    st.markdown(
        f'<div class="info-card" style="border-color:rgba(242,169,59,0.25);">'
        f'<h4 style="color:{C["accent"]};">Market Features + TDA Features + CatBoost</h4>'
        f'<p><strong>20 market features</strong>: Returns, log-returns, volatility (10/20-day), '
        f'momentum (5/10/20-day), MA distances, VIX level/change/ratio/MA.<br>'
        f'<strong>9 TDA features</strong>: H0 and H1 counts, total/max/mean persistence, H1 entropy.<br>'
        f'<strong>Total</strong>: 29 features.<br>'
        f'<strong>Model</strong>: CatBoost gradient boosting classifier with class-weight balancing.<br>'
        f'<strong>Evaluation</strong>: Temporal train/test split with out-of-sample backtesting.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Limitations
    section_header("Limitations & Caveats")
    limitations = [
        "Crash events are rare (~3.46% of observations). The model operates in a low base-rate environment where false positives are inherent.",
        "False positives remain significant. At the high-risk threshold (0.62), precision is ~20.65%, meaning ~4 out of 5 high-risk signals are false alarms.",
        "Predictions are probabilistic risk signals, not guarantees. Elevated probability indicates structural patterns associated with past crashes, not certainty of a future crash.",
        "Saved dashboard predictions are historical out-of-sample results. They are not live market signals unless a real-time inference pipeline is explicitly implemented.",
        "Financial markets can structurally change over time (regime shifts, policy changes, new instruments). Model assumptions may degrade.",
        "The model requires periodic retraining and revalidation to remain relevant.",
        "This is research and educational work, not financial advice. Do not use these predictions for actual trading decisions without independent professional review.",
    ]
    for lim in limitations:
        st.markdown(
            f'<div style="padding:8px 0 8px 16px; border-left:2px solid {C["copper"]}; margin-bottom:8px;">'
            f'<span style="color:{C["text2"]}; font-size:13px;">{lim}</span></div>',
            unsafe_allow_html=True,
        )

    # Download section
    section_header("Data Downloads")
    c1, c2, c3 = st.columns(3)
    download_files = [
        ("final_test_predictions.csv", "Predictions"),
        ("shap_feature_importance.csv", "SHAP Importance"),
        ("ablation_results.csv", "Ablation Results"),
    ]
    for i, (filename, label) in enumerate(download_files):
        col = [c1, c2, c3][i]
        with col:
            fpath = DATA_DIR / filename
            if fpath.exists():
                with open(fpath, "r") as f:
                    st.download_button(
                        f"Download {label}",
                        f.read(),
                        filename,
                        "text/csv",
                        key=f"dl_{filename}",
                    )


# ============================================================
# MAIN ROUTING
# ============================================================
def main():
    render_sidebar()

    # Load all data
    with st.spinner(""):
        data = load_all_data()

    page = st.session_state.get("page", "overview")

    page_map = {
        "overview": page_overview,
        "market": page_market,
        "tda": page_tda,
        "prediction": page_prediction,
        "explainability": page_explainability,
        "backtest": page_backtest,
        "performance": page_performance,
        "methodology": page_methodology,
    }

    handler = page_map.get(page, page_overview)
    handler(data)


if __name__ == "__main__":
    main()