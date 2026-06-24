import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
import textwrap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transformer Condition Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #07111f 0%, #0b1f36 50%, #07111f 100%);
}

[data-testid="stSidebar"] {
    background: #06101d;
    border-right: 1px solid rgba(0, 174, 255, 0.20);
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3, h4 {
    color: white;
}

p, label, span, div {
    color: #e9ecef;
}

/* Main top banner */
.small-banner {
    width: 100%;
    height: 350px;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(0, 174, 255, 0.35);
    box-shadow: 0 0 28px rgba(0, 174, 255, 0.18);
    margin-bottom: 24px;
    background-position: center center;
    background-repeat: no-repeat;
    background-size: cover;
}

/* General containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #081827;
    border-radius: 16px;
    border: 1px solid rgba(0, 174, 255, 0.25);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.38);
}

/* Inputs */
[data-testid="stNumberInput"] input {
    background-color: #20232d;
    color: white;
    border-radius: 8px;
}

[data-testid="stSelectbox"] div {
    color: white;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1971c2, #15aabf);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 0 18px rgba(0, 174, 255, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #15aabf, #1971c2);
    color: white;
}

/* ============================================================
   ATTRACTIVE PREDICTION RESULT SECTION
   ============================================================ */

.results-title-box {
    margin-top: 12px;
    margin-bottom: 18px;
    padding: 18px 22px;
    background: rgba(8, 24, 42, 0.94);
    border-radius: 16px;
    border: 1px solid rgba(0, 174, 255, 0.25);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.36);
}

.results-title-box h2 {
    margin: 0;
    font-size: 32px;
    color: #ffffff;
}

.results-title-box p {
    margin-top: 6px;
    margin-bottom: 0;
    color: #7dd3fc;
    font-size: 15px;
}

/* Result card */
.result-card {
    min-height: 365px;
    padding: 28px 30px;
    border-radius: 22px;
    background: #071827;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 35px rgba(0,0,0,0.45);
}

.result-card::before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: 0.18;
    background: radial-gradient(circle at top left, rgba(255,255,255,0.20), transparent 36%);
    pointer-events: none;
}

.result-card > * {
    position: relative;
    z-index: 2;
}

.result-header {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 24px;
}

.result-icon {
    width: 76px;
    height: 76px;
    min-width: 76px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.svg-icon {
    width: 40px;
    height: 40px;
    stroke: white;
    fill: none;
    stroke-width: 2.6;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.result-title h3 {
    margin: 0;
    font-size: 25px;
    line-height: 1.12;
    color: white;
    text-transform: uppercase;
}

.result-title span {
    font-size: 15px;
    font-weight: 700;
}

.result-value {
    font-size: 64px;
    font-weight: 900;
    margin-top: 20px;
    margin-bottom: 10px;
    letter-spacing: -1px;
}

.result-unit {
    font-size: 28px;
    color: #e9ecef;
    margin-left: 4px;
}

.condition-value-text {
    font-size: 44px;
    line-height: 1.15;
}

.result-pill {
    display: inline-block;
    margin-top: 14px;
    margin-bottom: 18px;
    padding: 8px 24px;
    border-radius: 999px;
    font-size: 20px;
    font-weight: 900;
    letter-spacing: 0.4px;
}

.result-description {
    font-size: 15px;
    line-height: 1.45;
    color: #d0d7de;
    margin-top: 8px;
}

/* Health card */
.health-card {
    border: 1px solid rgba(0, 174, 255, 0.88);
    box-shadow: 0 0 26px rgba(0, 174, 255, 0.25);
    background: linear-gradient(145deg, rgba(4, 26, 48, 0.98), rgba(7, 24, 39, 0.96));
}

.health-card .result-icon {
    background: linear-gradient(135deg, #0b5ed7, #22d3ee);
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.40);
}

.health-card .result-title span,
.health-card .result-value {
    color: #38d9ff;
}

.health-card .result-pill {
    color: #38d9ff;
    border: 1px solid rgba(56, 217, 255, 0.75);
    background: rgba(56, 217, 255, 0.08);
}

/* Paper card */
.paper-card {
    border: 1px solid rgba(250, 176, 5, 0.90);
    box-shadow: 0 0 28px rgba(250, 176, 5, 0.30);
    background: linear-gradient(145deg, rgba(36, 25, 4, 0.98), rgba(8, 24, 39, 0.96));
}

.paper-card .result-icon {
    background: linear-gradient(135deg, #e67700, #fab005);
    box-shadow: 0 0 22px rgba(250, 176, 5, 0.45);
}

.paper-card .result-title span,
.paper-card .result-value {
    color: #fab005;
}

.paper-card .result-pill {
    color: #ffffff;
    border: 1px solid rgba(250, 176, 5, 0.85);
    background: rgba(250, 176, 5, 0.20);
}

/* Transformer condition cards */
.condition-good {
    border: 1px solid rgba(64, 192, 87, 0.90);
    box-shadow: 0 0 28px rgba(64, 192, 87, 0.25);
    background: linear-gradient(145deg, rgba(5, 35, 18, 0.98), rgba(8, 24, 39, 0.96));
}

.condition-good .result-icon {
    background: linear-gradient(135deg, #2f9e44, #69db7c);
}

.condition-good .result-title span,
.condition-good .result-value {
    color: #69db7c;
}

.condition-good .result-pill {
    color: #69db7c;
    border: 1px solid rgba(105, 219, 124, 0.75);
    background: rgba(105, 219, 124, 0.10);
}

.condition-fair {
    border: 1px solid rgba(77, 171, 247, 0.90);
    box-shadow: 0 0 28px rgba(77, 171, 247, 0.25);
    background: linear-gradient(145deg, rgba(4, 26, 48, 0.98), rgba(8, 24, 39, 0.96));
}

.condition-fair .result-icon {
    background: linear-gradient(135deg, #1971c2, #4dabf7);
}

.condition-fair .result-title span,
.condition-fair .result-value {
    color: #4dabf7;
}

.condition-fair .result-pill {
    color: #4dabf7;
    border: 1px solid rgba(77, 171, 247, 0.75);
    background: rgba(77, 171, 247, 0.10);
}

.condition-warning {
    border: 1px solid rgba(255, 146, 43, 0.92);
    box-shadow: 0 0 28px rgba(255, 146, 43, 0.30);
    background: linear-gradient(145deg, rgba(44, 24, 4, 0.98), rgba(8, 24, 39, 0.96));
}

.condition-warning .result-icon {
    background: linear-gradient(135deg, #e67700, #ff922b);
}

.condition-warning .result-title span,
.condition-warning .result-value {
    color: #ff922b;
}

.condition-warning .result-pill {
    color: #ff922b;
    border: 1px solid rgba(255, 146, 43, 0.75);
    background: rgba(255, 146, 43, 0.10);
}

.condition-critical {
    border: 1px solid rgba(255, 107, 107, 0.95);
    box-shadow: 0 0 28px rgba(255, 107, 107, 0.32);
    background: linear-gradient(145deg, rgba(48, 8, 8, 0.98), rgba(8, 24, 39, 0.96));
}

.condition-critical .result-icon {
    background: linear-gradient(135deg, #c92a2a, #ff6b6b);
}

.condition-critical .result-title span,
.condition-critical .result-value {
    color: #ff6b6b;
}

.condition-critical .result-pill {
    color: #ff6b6b;
    border: 1px solid rgba(255, 107, 107, 0.75);
    background: rgba(255, 107, 107, 0.10);
}

/* Meaning panel */
.meaning-panel {
    margin-top: 24px;
    padding: 22px 28px;
    border-radius: 18px;
    background: #081827;
    border: 1px solid rgba(0, 174, 255, 0.25);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.38);
}

.meaning-panel h3 {
    margin: 0 0 16px 0;
    font-size: 24px;
    color: #38d9ff;
}

.meaning-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 22px;
}

.meaning-item {
    display: flex;
    gap: 14px;
    align-items: flex-start;
}

.meaning-icon {
    min-width: 52px;
    height: 52px;
    border-radius: 50%;
    background: rgba(56, 217, 255, 0.12);
    border: 1px solid rgba(56, 217, 255, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
}

.meaning-icon svg {
    width: 28px;
    height: 28px;
}

.meaning-item h4 {
    margin: 0;
    font-size: 17px;
}

.meaning-item p {
    margin: 4px 0 0 0;
    font-size: 14px;
    line-height: 1.45;
    color: #cbd5e1;
}

.decision-note {
    margin-top: 16px;
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(25, 113, 194, 0.12);
    border: 1px solid rgba(77, 171, 247, 0.28);
    font-size: 14px;
    color: #dbeafe;
}

/* Diagnostic title */
.diagnostic-title-box {
    margin-top: 14px;
    margin-bottom: 18px;
    padding: 18px 22px;
    background: rgba(8, 24, 42, 0.94);
    border-radius: 16px;
    border: 1px solid rgba(0, 174, 255, 0.25);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.36);
}

.diagnostic-title-box h2 {
    margin: 0;
    font-size: 30px;
    color: #ffffff;
}

.diagnostic-title-box p {
    margin-top: 6px;
    margin-bottom: 0;
    color: #7dd3fc;
    font-size: 15px;
}

.footer-custom {
    text-align: center;
    color: #adb5bd;
    font-size: 13px;
    padding: 22px;
    margin-top: 35px;
    border-top: 1px solid rgba(255,255,255,0.15);
    background: #081827;
    border-radius: 14px;
}

@media (max-width: 1100px) {
    .meaning-grid {
        grid-template-columns: 1fr;
    }

    .result-card {
        min-height: auto;
        margin-bottom: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    model_path = "transformer_extra_trees_model_compressed.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload transformer_extra_trees_model.pkl")
        st.stop()

    model_package = joblib.load(model_path)

    if isinstance(model_package, dict):
        if "regression_model" in model_package:
            model = model_package["regression_model"]
        elif "model" in model_package:
            model = model_package["model"]
        else:
            st.error("Model package does not contain regression_model or model key.")
            st.stop()

        feature_names = model_package.get("feature_names", None)

    else:
        model = model_package
        feature_names = None

    return model, feature_names


model, feature_names = load_model()


# ============================================================
# HTML HELPER FUNCTIONS
# ============================================================

def clean_html(raw_html):
    return " ".join(textwrap.dedent(raw_html).strip().split())


def render_html(raw_html):
    st.markdown(clean_html(raw_html), unsafe_allow_html=True)


# ============================================================
# SVG ICONS
# ============================================================

def health_svg():
    return """
    <svg class="svg-icon" viewBox="0 0 24 24">
        <path d="M20.8 4.6c-1.8-1.8-4.7-1.8-6.5 0L12 6.9 9.7 4.6c-1.8-1.8-4.7-1.8-6.5 0s-1.8 4.7 0 6.5L12 20l8.8-8.9c1.8-1.8 1.8-4.7 0-6.5z"></path>
        <path d="M7 12h3l1.2-2.5L14 15l1.4-3H18"></path>
    </svg>
    """


def paper_svg():
    return """
    <svg class="svg-icon" viewBox="0 0 24 24">
        <path d="M7 3h7l5 5v13H7z"></path>
        <path d="M14 3v5h5"></path>
        <path d="M10 13h6"></path>
        <path d="M10 17h6"></path>
    </svg>
    """


def shield_svg():
    return """
    <svg class="svg-icon" viewBox="0 0 24 24">
        <path d="M12 3l7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6z"></path>
        <path d="M9 12l2 2 4-4"></path>
    </svg>
    """


def warning_svg():
    return """
    <svg class="svg-icon" viewBox="0 0 24 24">
        <path d="M12 3l10 18H2z"></path>
        <path d="M12 9v5"></path>
        <path d="M12 17h.01"></path>
    </svg>
    """


def info_svg():
    return """
    <svg class="svg-icon" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="9"></circle>
        <path d="M12 11v6"></path>
        <path d="M12 7h.01"></path>
    </svg>
    """


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


background_path = "assets/transformer_banner.png"

if os.path.exists(background_path):
    background_base64 = image_to_base64(background_path)

    render_html(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(3, 12, 24, 0.80), rgba(3, 12, 24, 0.88)),
            url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    </style>
    """)

else:
    st.warning("Background image not found. Please check assets/transformer_banner.png")


# ============================================================
# MODEL HELPER FUNCTIONS
# ============================================================

def classify_condition(hi):
    if hi >= 85:
        return "Good"
    elif hi >= 70:
        return "Fair"
    elif hi >= 50:
        return "Poor / Warning"
    else:
        return "Critical"


def classify_paper_condition(dp):
    if dp > 800:
        return "Healthy Paper"
    elif dp > 500:
        return "Moderate Ageing"
    elif dp > 250:
        return "Aged / Warning"
    elif dp > 150:
        return "Severe Ageing"
    else:
        return "Critical Paper Ageing"


def get_condition_card_class(condition):
    if condition == "Good":
        return "condition-good"
    elif condition == "Fair":
        return "condition-fair"
    elif condition == "Poor / Warning":
        return "condition-warning"
    else:
        return "condition-critical"


def get_condition_icon_svg(condition):
    if condition == "Good":
        return shield_svg()
    elif condition == "Fair":
        return info_svg()
    elif condition == "Poor / Warning":
        return warning_svg()
    else:
        return warning_svg()


def get_condition_sentence(condition):
    if condition == "Good":
        return "Transformer is predicted to be in healthy operating condition."
    elif condition == "Fair":
        return "Transformer is acceptable, but regular monitoring is recommended."
    elif condition == "Poor / Warning":
        return "Transformer shows warning signs and needs attention."
    else:
        return "Transformer is in critical condition and needs urgent expert investigation."


def create_features(input_data):
    df = pd.DataFrame([input_data])

    df["TDCG"] = (
        df["H2"]
        + df["Methane"]
        + df["Ethane"]
        + df["Ethylene"]
        + df["Acetylene"]
        + df["CO"]
    )

    df["EstimatedDP"] = np.where(
        df["Furan"] > 0,
        (1.51 - np.log10(df["Furan"])) / 0.0035,
        1000
    )

    df["EstimatedDP"] = df["EstimatedDP"].clip(lower=0, upper=1200)
    df["CO2_CO_Ratio"] = df["CO2"] / (df["CO"] + 1e-6)

    df["Furan_Acid"] = df["Furan"] * df["Acid"]
    df["Water_Acid"] = df["Water"] * df["Acid"]
    df["Inv_BDV"] = 1 / (df["BDV"] + 1e-6)
    df["Inv_IFT"] = 1 / (df["IFT"] + 1e-6)

    log_columns = [
        "H2", "CO", "CO2", "Methane", "Acetylene",
        "Ethylene", "Ethane", "Furan", "TDCG"
    ]

    for col in log_columns:
        df["log_" + col] = np.log10(df[col] + 1)

    return df


def predict_hi(input_data):
    df_features = create_features(input_data)

    if feature_names is not None:
        missing_features = [f for f in feature_names if f not in df_features.columns]

        if len(missing_features) > 0:
            st.error(f"Missing required features: {missing_features}")
            st.stop()

        X_new = df_features[feature_names]
    else:
        X_new = df_features

    predicted_hi = model.predict(X_new)[0]
    predicted_hi = float(np.clip(predicted_hi, 0, 100))

    return predicted_hi, df_features


# ============================================================
# PLOTS
# ============================================================

def plot_hi_gauge(predicted_hi):
    condition = classify_condition(predicted_hi)

    if condition == "Good":
        condition_color = "#69db7c"
    elif condition == "Fair":
        condition_color = "#4dabf7"
    elif condition == "Poor / Warning":
        condition_color = "#fab005"
    else:
        condition_color = "#ff6b6b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_hi,
        number={
            "font": {"size": 44, "color": "white"},
            "valueformat": ".1f"
        },
        title={
            "text": "<b>HEALTH INDEX GAUGE</b>",
            "font": {"size": 16, "color": "#7dd3fc"}
        },
        gauge={
            "shape": "angular",
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
                "tickwidth": 1,
                "tickcolor": "white",
                "tickfont": {"size": 13, "color": "white"}
            },
            "bar": {
                "color": "rgba(0,0,0,0)"
            },
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#e03131"},
                {"range": [25, 50], "color": "#f76707"},
                {"range": [50, 75], "color": "#fab005"},
                {"range": [75, 100], "color": "#2f9e44"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 5},
                "thickness": 0.75,
                "value": predicted_hi
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=70, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        annotations=[
            dict(
                x=0.5,
                y=0.03,
                text=f"<b>{condition}</b>",
                showarrow=False,
                font=dict(size=18, color=condition_color)
            )
        ]
    )

    return fig


def plot_dga_gases(input_data):
    gas_names = ["H₂", "CH₄", "C₂H₆", "C₂H₄", "C₂H₂", "CO", "CO₂"]

    gas_values = [
        input_data["H2"],
        input_data["Methane"],
        input_data["Ethane"],
        input_data["Ethylene"],
        input_data["Acetylene"],
        input_data["CO"],
        input_data["CO2"]
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gas_names,
        y=gas_values,
        text=[f"{v:.1f}" for v in gas_values],
        textposition="outside",
        marker=dict(color="#4dabf7")
    ))

    fig.update_layout(
        title="DGA Gas Concentrations",
        xaxis_title="Gas Type",
        yaxis_title="Concentration (ppm)",
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def plot_hi_condition_scale(predicted_hi):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[50],
        y=["Health Index"],
        orientation="h",
        name="Critical",
        marker=dict(color="#c92a2a"),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Bar(
        x=[20],
        y=["Health Index"],
        orientation="h",
        name="Poor / Warning",
        marker=dict(color="#f08c00"),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Bar(
        x=[15],
        y=["Health Index"],
        orientation="h",
        name="Fair",
        marker=dict(color="#1971c2"),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Bar(
        x=[15],
        y=["Health Index"],
        orientation="h",
        name="Good",
        marker=dict(color="#2f9e44"),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=[predicted_hi],
        y=["Health Index"],
        mode="markers+text",
        marker=dict(size=18, color="white"),
        text=[f"HI = {predicted_hi:.2f}"],
        textposition="top center",
        name="Predicted HI"
    ))

    fig.update_layout(
        barmode="stack",
        title="Health Index Condition Scale",
        xaxis=dict(range=[0, 100], title="Health Index"),
        template="plotly_dark",
        height=220,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ Model Information")

st.sidebar.markdown("""
**Project:** Transformer Condition Monitoring  
**Model:** Extra Trees Regressor  
**Target:** Health Index Prediction  
**Output Range:** 0–100
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("R² Score", "0.9500")
st.sidebar.metric("MAE", "2.6520")
st.sidebar.metric("RMSE", "4.6087")
st.sidebar.metric("Condition Accuracy", "91.78%")

st.sidebar.markdown("---")

st.sidebar.subheader("🚦 Condition Limits")

st.sidebar.markdown("""
| HI Range | Condition |
|---|---|
| 85–100 | Good |
| 70–85 | Fair |
| 50–70 | Poor / Warning |
| < 50 | Critical |
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "Academic decision-support tool. Final decisions should be made with expert engineering assessment."
)


# ============================================================
# TOP BANNER
# ============================================================

banner_path = "assets/top_banner.png"

if os.path.exists(banner_path):
    banner_base64 = image_to_base64(banner_path)

    render_html(f"""
    <div class="small-banner"
         style="background-image:
         linear-gradient(rgba(3, 12, 24, 0.00), rgba(3, 12, 24, 0.05)),
         url('data:image/png;base64,{banner_base64}');">
    </div>
    """)

else:
    st.warning("Top banner image not found. Please check assets/top_banner.png")


# ============================================================
# PRESET SAMPLE CASES
# ============================================================

preset_cases = {
    "Dataset Example": {
        "H2": 11.8,
        "CO": 244.4,
        "CO2": 3032.4,
        "Methane": 0.0,
        "Acetylene": 0.0,
        "Ethylene": 1.0,
        "Ethane": 1.4,
        "Furan": 1.03,
        "Water": 13.0,
        "Acid": 0.005,
        "BDV": 48.0,
        "DDF1": 0.038,
        "DDF2": 0.004,
        "Color": 2.0,
        "IFT": 24.0
    },
    "Healthy Example": {
        "H2": 5.0,
        "CO": 100.0,
        "CO2": 1000.0,
        "Methane": 2.0,
        "Acetylene": 0.0,
        "Ethylene": 1.0,
        "Ethane": 1.0,
        "Furan": 0.01,
        "Water": 5.0,
        "Acid": 0.01,
        "BDV": 75.0,
        "DDF1": 0.005,
        "DDF2": 0.001,
        "Color": 0.5,
        "IFT": 40.0
    },
    "Poor / Warning Example": {
        "H2": 80.0,
        "CO": 800.0,
        "CO2": 9000.0,
        "Methane": 80.0,
        "Acetylene": 2.0,
        "Ethylene": 50.0,
        "Ethane": 60.0,
        "Furan": 2.0,
        "Water": 20.0,
        "Acid": 0.08,
        "BDV": 35.0,
        "DDF1": 0.05,
        "DDF2": 0.01,
        "Color": 3.0,
        "IFT": 18.0
    },
    "Critical Example": {
        "H2": 300.0,
        "CO": 1200.0,
        "CO2": 15000.0,
        "Methane": 250.0,
        "Acetylene": 40.0,
        "Ethylene": 300.0,
        "Ethane": 200.0,
        "Furan": 8.0,
        "Water": 35.0,
        "Acid": 0.2,
        "BDV": 18.0,
        "DDF1": 0.12,
        "DDF2": 0.03,
        "Color": 4.0,
        "IFT": 12.0
    }
}

st.subheader("Select Sample Case or Enter New Values")

selected_case = st.selectbox(
    "Preset sample case",
    list(preset_cases.keys())
)

default_values = preset_cases[selected_case]


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Input Parameters")

input_col1, input_col2, input_col3 = st.columns([1.35, 0.9, 1.25])

with input_col1:
    with st.container(border=True):
        st.markdown("### DGA Gases")
        c1, c2 = st.columns(2)

        with c1:
            H2 = st.number_input(
                "Hydrogen H₂ (ppm)",
                min_value=0.0,
                value=float(default_values["H2"]),
                help="Hydrogen gas may increase due to partial discharge or electrical stress."
            )

            Methane = st.number_input(
                "Methane CH₄ (ppm)",
                min_value=0.0,
                value=float(default_values["Methane"]),
                help="Methane can indicate low-temperature thermal faults or oil decomposition."
            )

            Ethylene = st.number_input(
                "Ethylene C₂H₄ (ppm)",
                min_value=0.0,
                value=float(default_values["Ethylene"]),
                help="Ethylene is usually related to higher temperature thermal faults."
            )

            Ethane = st.number_input(
                "Ethane C₂H₆ (ppm)",
                min_value=0.0,
                value=float(default_values["Ethane"]),
                help="Ethane can be related to low-temperature overheating."
            )

        with c2:
            Acetylene = st.number_input(
                "Acetylene C₂H₂ (ppm)",
                min_value=0.0,
                value=float(default_values["Acetylene"]),
                help="Acetylene is commonly associated with arcing or high-energy electrical faults."
            )

            CO = st.number_input(
                "Carbon Monoxide CO (ppm)",
                min_value=0.0,
                value=float(default_values["CO"]),
                help="CO is related to cellulose paper insulation degradation."
            )

            CO2 = st.number_input(
                "Carbon Dioxide CO₂ (ppm)",
                min_value=0.0,
                value=float(default_values["CO2"]),
                help="CO₂ is also related to paper insulation ageing and thermal degradation."
            )

with input_col2:
    with st.container(border=True):
        st.markdown("### Furan")

        Furan = st.number_input(
            "2-FAL / Furan (ppm)",
            min_value=0.0,
            value=float(default_values["Furan"]),
            help="Furan is used as an indirect indicator of cellulose paper insulation ageing."
        )

with input_col3:
    with st.container(border=True):
        st.markdown("### Oil Quality")
        c1, c2 = st.columns(2)

        with c1:
            Water = st.number_input(
                "Moisture / Water Content (ppm)",
                min_value=0.0,
                value=float(default_values["Water"]),
                help="Water content in transformer oil. Higher moisture reduces dielectric strength."
            )

            Acid = st.number_input(
                "Acidity / Neutralization Number (mg KOH/g)",
                min_value=0.0,
                value=float(default_values["Acid"]),
                format="%.4f",
                help="Acidity is measured in mg KOH/g oil. Higher value indicates oil oxidation and ageing."
            )

            BDV = st.number_input(
                "Breakdown Voltage, BDV (kV)",
                min_value=0.0,
                value=float(default_values["BDV"]),
                help="BDV represents the dielectric strength of transformer oil. Higher value usually indicates better insulation quality."
            )

        with c2:
            DDF1 = st.number_input(
                "DDF1 / tan δ (dimensionless)",
                min_value=0.0,
                value=float(default_values["DDF1"]),
                format="%.4f",
                help="Dielectric Dissipation Factor. Higher value indicates higher dielectric loss or contamination."
            )

            DDF2 = st.number_input(
                "DDF2 / tan δ (dimensionless)",
                min_value=0.0,
                value=float(default_values["DDF2"]),
                format="%.4f",
                help="Second dielectric dissipation factor measurement. Treated as a dimensionless dielectric loss indicator."
            )

            Color = st.number_input(
                "Oil Color Index (ASTM D1500)",
                min_value=0.0,
                max_value=8.0,
                value=float(default_values["Color"]),
                step=0.5,
                help="1 = pale yellow, 2 = yellow, 3 = amber, 4 = dark brown"
            )

            IFT = st.number_input(
                "Interfacial Tension, IFT (mN/m)",
                min_value=0.0,
                value=float(default_values["IFT"]),
                help="Higher IFT indicates cleaner oil. Lower IFT indicates oxidation products or contamination."
            )


# ============================================================
# PREDICTION SECTION
# ============================================================

input_data = {
    "H2": H2,
    "CO": CO,
    "CO2": CO2,
    "Methane": Methane,
    "Acetylene": Acetylene,
    "Ethylene": Ethylene,
    "Ethane": Ethane,
    "Furan": Furan,
    "Water": Water,
    "Acid": Acid,
    "BDV": BDV,
    "DDF1": DDF1,
    "DDF2": DDF2,
    "Color": Color,
    "IFT": IFT
}

st.markdown("---")

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    predict_button = st.button(
        "⚡ Predict Transformer Condition",
        use_container_width=True
    )


if predict_button:

    predicted_hi, df_features = predict_hi(input_data)

    tdcg = float(df_features["TDCG"].iloc[0])
    estimated_dp = float(df_features["EstimatedDP"].iloc[0])
    co2_co_ratio = float(df_features["CO2_CO_Ratio"].iloc[0])

    transformer_condition = classify_condition(predicted_hi)
    paper_condition = classify_paper_condition(estimated_dp)
    condition_class = get_condition_card_class(transformer_condition)
    condition_icon_svg = get_condition_icon_svg(transformer_condition)
    condition_sentence = get_condition_sentence(transformer_condition)

    # ========================================================
    # ATTRACTIVE RESULT CARDS
    # ========================================================

    render_html("""
    <div class="results-title-box">
        <h2>📈 Prediction Results</h2>
        <p>Machine learning assessment based on DGA, furan analysis and oil quality parameters</p>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        render_html(f"""
        <div class="result-card health-card">
            <div class="result-header">
                <div class="result-icon">{health_svg()}</div>
                <div class="result-title">
                    <h3>Predicted<br>Health Index</h3>
                    <span>Overall Equipment Health</span>
                </div>
            </div>
            <div class="result-value">{predicted_hi:.1f}<span class="result-unit">/100</span></div>
            <div class="result-pill">{transformer_condition.upper()}</div>
            <p class="result-description">
                Main health score of the transformer. A higher value means a healthier transformer condition.
            </p>
        </div>
        """)

    with col2:
        render_html(f"""
        <div class="result-card paper-card">
            <div class="result-header">
                <div class="result-icon">{paper_svg()}</div>
                <div class="result-title">
                    <h3>Paper Insulation</h3>
                    <span>Degree of Polymerization (DP)</span>
                </div>
            </div>
            <div class="result-value">DP = {estimated_dp:.0f}</div>
            <div class="result-pill">{paper_condition.upper()}</div>
            <p class="result-description">
                Lower DP means the cellulose paper insulation is more aged and mechanically weaker.
            </p>
        </div>
        """)

    with col3:
        render_html(f"""
        <div class="result-card {condition_class}">
            <div class="result-header">
                <div class="result-icon">{condition_icon_svg}</div>
                <div class="result-title">
                    <h3>Transformer<br>Condition</h3>
                    <span>Final Assessment</span>
                </div>
            </div>
            <div class="result-value condition-value-text">{transformer_condition}</div>
            <div class="result-pill">FINAL DECISION</div>
            <p class="result-description">
                {condition_sentence}
            </p>
        </div>
        """)

    # ========================================================
    # MEANING PANEL
    # ========================================================

    render_html(f"""
    <div class="meaning-panel">
        <h3>💡 What do these results mean?</h3>
        <div class="meaning-grid">
            <div class="meaning-item">
                <div class="meaning-icon">{health_svg()}</div>
                <div>
                    <h4 style="color:#38d9ff;">Health Index</h4>
                    <p>Overall health score predicted by the machine learning model using all input parameters.</p>
                </div>
            </div>
            <div class="meaning-item">
                <div class="meaning-icon">{paper_svg()}</div>
                <div>
                    <h4 style="color:#fab005;">Paper Insulation</h4>
                    <p>Shows the ageing level of cellulose paper insulation using Degree of Polymerization.</p>
                </div>
            </div>
            <div class="meaning-item">
                <div class="meaning-icon">{shield_svg()}</div>
                <div>
                    <h4 style="color:#69db7c;">Transformer Condition</h4>
                    <p>Final category such as Good, Fair, Poor / Warning, or Critical.</p>
                </div>
            </div>
        </div>
        <div class="decision-note">
            ℹ️ This is a decision-support tool. Final maintenance decisions should be made with expert engineering assessment and standard diagnostic practices.
        </div>
    </div>
    """)

    # ========================================================
    # DIAGNOSTIC DASHBOARD
    # ========================================================

    render_html("""
    <div class="diagnostic-title-box">
        <h2>📊 Diagnostic Dashboard</h2>
        <p>Supporting visual analysis for health index, gas generation and condition range</p>
    </div>
    """)

    dash_col1, dash_col2 = st.columns([1, 1.35])

    with dash_col1:
        with st.container(border=True):
            st.plotly_chart(
                plot_hi_gauge(predicted_hi),
                use_container_width=True
            )

    with dash_col2:
        with st.container(border=True):
            st.plotly_chart(
                plot_dga_gases(input_data),
                use_container_width=True
            )

    with st.container(border=True):
        st.plotly_chart(
            plot_hi_condition_scale(predicted_hi),
            use_container_width=True
        )

    with st.expander("View Entered Input Data"):
        input_table = pd.DataFrame([input_data]).T
        input_table.columns = ["Value"]
        st.dataframe(input_table, use_container_width=True)

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    report_data = {
        "Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Health Index": predicted_hi,
        "Transformer Condition": transformer_condition,
        "TDCG": tdcg,
        "Estimated DP": estimated_dp,
        "Degree of Polymerization": estimated_dp,
        "Paper Condition": paper_condition,
        "CO2/CO Ratio": co2_co_ratio,
        **input_data
    }

    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Report as CSV",
        data=csv,
        file_name="transformer_condition_report.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer-custom">
    Developed for Undergraduate High Voltage Engineering Project<br>
    Transformer Condition Monitoring using DGA, Furan, Oil Quality Data and Machine Learning<br>
    This system is a decision-support tool and should be used together with standard diagnostic practices and expert assessment.
</div>
""")