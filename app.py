"""
============================================================
  Laptop Price Predictor — Streamlit Application
  Author  : Akash Awasthi
  Version : 2.0
============================================================
  Run locally:
      streamlit run app.py

  Requires:
      pip install streamlit pandas joblib scikit-learn numpy
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS  — minimal dark-card aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar background */
    [data-testid="stSidebar"] {background-color: #1e1e2e;}

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #2a2a3e;
        border: 1px solid #3d3d5c;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 2em;
        font-size: 1.05em;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {opacity: 0.85;}

    /* Section headers */
    .section-header {
        color: #a78bfa;
        font-size: 1.1em;
        font-weight: 700;
        border-bottom: 2px solid #3d3d5c;
        padding-bottom: 4px;
        margin-bottom: 14px;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #667eea;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }
    .result-price {
        font-size: 2.8em;
        font-weight: 800;
        color: #a78bfa;
    }
    .result-label {
        color: #94a3b8;
        font-size: 1em;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
EURO_TO_INR = 90.0
EURO_TO_USD = 1.08

# ─────────────────────────────────────────────────────────────────────────────
#  Load Model Bundle
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    """
    Load the serialised sklearn Pipeline + metadata from disk.
    Cached with @st.cache_resource so it loads only once per session.
    """
    base_dir   = Path(__file__).resolve().parent
    model_path = base_dir / "model" / "laptop_price_model.pkl"

    if not model_path.exists():
        st.error(
            f"⚠️  Model file not found at `{model_path}`.\n\n"
            "Please run `python laptopPricePredictorAPI.py` first to train and save the model."
        )
        st.stop()

    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"], bundle.get("metrics", {}), bundle.get("model_name", "ML Model")


model, feature_order, train_metrics, model_name = load_model_bundle()

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — Model Info & Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💻 Laptop Price Predictor")
    st.markdown("---")

    st.markdown("### 🤖 Model Info")
    st.info(f"**Algorithm:** {model_name}")

    if train_metrics:
        st.markdown("### 📊 Training Metrics")
        col_a, col_b = st.columns(2)
        col_a.metric("R²",      f"{train_metrics.get('R2',   0):.4f}")
        col_b.metric("Adj-R²",  f"{train_metrics.get('Adj_R2', 0):.4f}")
        col_a.metric("MAE",     f"€ {train_metrics.get('MAE',  0):.0f}")
        col_b.metric("RMSE",    f"€ {train_metrics.get('RMSE', 0):.0f}")

    st.markdown("---")
    st.markdown("### 💱 Currency Rates")
    euro_inr = st.number_input("€ → ₹ rate", value=EURO_TO_INR, step=1.0, format="%.1f")
    euro_usd = st.number_input("€ → $ rate", value=EURO_TO_USD, step=0.01, format="%.2f")

    st.markdown("---")
    st.caption("End-to-End ML Project  \nEDA · Outlier Removal · GridSearchCV · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
#  Main — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 💻 Laptop Price Prediction System")
st.markdown(
    "Fill in the laptop specifications below and click **Predict Price** "
    "to get an instant AI-powered price estimate."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Input Form  — 3-column layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Row 1: Brand & Type ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔹 Brand & Type</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox(
        "Brand / Company",
        ["Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Toshiba",
         "Apple", "Samsung", "Razer", "Microsoft", "Xiaomi", "LG", "Huawei", "Other"],
        help="Laptop manufacturer"
    )

with col2:
    type_name = st.selectbox(
        "Laptop Type",
        ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible", "Workstation", "Netbook"],
        help="Primary use-case / form-factor"
    )

with col3:
    os = st.selectbox(
        "Operating System",
        ["Windows", "macOS", "Linux", "Other"],
        help="Pre-installed OS category"
    )

# ── Row 2: Hardware ───────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">🔹 Hardware</p>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    ram = st.selectbox(
        "RAM (GB)",
        [2, 4, 6, 8, 12, 16, 24, 32, 64],
        index=3,  # default 8GB
        help="System RAM in gigabytes"
    )

with col2:
    cpu_brand = st.selectbox(
        "CPU Brand",
        ["Intel", "AMD", "Samsung"],
        help="Processor manufacturer"
    )

with col3:
    gpu_brand = st.selectbox(
        "GPU Brand",
        ["Intel", "Nvidia", "AMD"],
        help="Graphics card manufacturer"
    )

with col4:
    weight = st.number_input(
        "Weight (kg)",
        min_value=0.5, max_value=5.0,
        value=1.8, step=0.1,
        help="Laptop weight in kilograms"
    )

# ── Row 3: Storage & Display ──────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">🔹 Storage & Display</p>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    hdd = st.selectbox("HDD", [0, 1], format_func=lambda x: "Yes" if x else "No",
                       help="Has HDD storage?")

with col2:
    ssd = st.selectbox("SSD", [1, 0], format_func=lambda x: "Yes" if x else "No",
                       help="Has SSD storage?")

with col3:
    ppi = st.number_input(
        "Display PPI",
        min_value=80.0, max_value=400.0,
        value=141.0, step=1.0,
        help="Pixels Per Inch — higher = sharper display"
    )

with col4:
    touchscreen = st.selectbox("Touchscreen", [0, 1],
                               format_func=lambda x: "Yes" if x else "No",
                               help="Does it have a touchscreen?")

with col5:
    ips = st.selectbox("IPS Panel", [1, 0],
                       format_func=lambda x: "Yes" if x else "No",
                       help="IPS display for wider colour gamut?")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────────────────────
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔍  Predict Price")

if predict_btn:
    # ── Build input DataFrame in the exact column order used during training ──
    input_data = {
        "Company"   : company,
        "TypeName"  : type_name,
        "Ram"       : int(ram),
        "Weight"    : float(weight),
        "Touchscreen": int(touchscreen),
        "Ips"       : int(ips),
        "ppi"       : float(ppi),
        "Cpu brand" : cpu_brand,
        "HDD"       : int(hdd),
        "SSD"       : int(ssd),
        "Gpu brand" : gpu_brand,
        "os"        : os,
    }

    input_df = pd.DataFrame([input_data])[feature_order]

    # ── Predict ───────────────────────────────────────────────────────────────
    price_eur = float(model.predict(input_df)[0])
    price_inr = price_eur * euro_inr
    price_usd = price_eur * euro_usd

    # ── Result Display ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Estimated Laptop Price</div>
            <div class="result-price">₹ {int(price_inr):,}</div>
            <div class="result-label" style="margin-top:8px;">
                € {price_eur:,.0f} &nbsp;|&nbsp; $ {price_usd:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Spec Summary ──────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("#### 📋 Specification Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(f"- **Brand:** {company}  |  **Type:** {type_name}")
        st.markdown(f"- **OS:** {os}  |  **Weight:** {weight} kg")
        st.markdown(f"- **RAM:** {ram} GB  |  **CPU:** {cpu_brand}  |  **GPU:** {gpu_brand}")

    with summary_col2:
        st.markdown(f"- **Storage:** {'SSD ✔' if ssd else 'SSD ✗'}  {'HDD ✔' if hdd else 'HDD ✗'}")
        st.markdown(f"- **Display PPI:** {ppi:.0f}  |  **IPS:** {'Yes' if ips else 'No'}  |  **Touch:** {'Yes' if touchscreen else 'No'}")
        st.markdown(f"- **Model used:** `{model_name}`")

    # ── Price confidence hint ─────────────────────────────────────────────────
    mae = train_metrics.get("MAE", 150)
    st.info(
        f"ℹ️  Model MAE is **€ {mae:.0f}** — the actual price typically falls within "
        f"± ₹ {int(mae * euro_inr):,} of this estimate."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Scikit-learn · Streamlit · Python  |  "
    "Dataset: 1,303 laptops · Features: Brand, RAM, CPU, GPU, Storage, Display  |  "
    f"Best Model: {model_name}"
)
