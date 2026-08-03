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

from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
import requests

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
    /* Global background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid #334155;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Hero card */
    .hero-box {
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(59,130,246,0.22));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 18px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.18);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        color: white;
        font-size: 0.8em;
        font-weight: 700;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 1.6em;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 0.98em;
        line-height: 1.5;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.16));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 12px;
        padding: 12px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7em 1.8em;
        font-size: 1.02em;
        font-weight: 700;
        width: 100%;
        transition: 0.25s ease;
        box-shadow: 0 8px 20px rgba(59,130,246,0.25);
    }
    .stButton > button:hover {opacity: 0.92; transform: translateY(-1px);}

    /* Section headers */
    .section-header {
        color: #8b5cf6;
        font-size: 1.1em;
        font-weight: 700;
        border-bottom: 2px solid rgba(139,92,246,0.35);
        padding-bottom: 4px;
        margin-bottom: 14px;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #111827, #1f2937);
        border: 1px solid rgba(96,165,250,0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    }
    .result-price {
        font-size: 2.8em;
        font-weight: 800;
        color: #f8fafc;
    }
    .result-label {
        color: #cbd5e1;
        font-size: 1em;
    }

    /* Image cards */
    .image-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.16);
        margin-top: 6px;
    }

    /* Form stability */
    .stSelectbox > label, .stNumberInput > label, .stTextInput > label {
        color: #e2e8f0;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stWidgetLabel"] {
        margin-bottom: 0.25rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    /* Brand grid */
    .brand-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 16px;
        margin-top: 16px;
        align-items: center;
    }
    .brand-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 12px 10px;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(148,163,184,0.14);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .brand-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 32px rgba(0,0,0,0.24);
        background: rgba(255,255,255,0.09);
    }
    .brand-label { color: #cbd5e1; font-size: 0.95rem; margin-top: 4px; }
    .form-card {
        background: rgba(15, 23, 42, 0.84);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 24px;
        padding: 26px 28px;
        box-shadow: 0 18px 42px rgba(0,0,0,0.25);
        margin-bottom: 24px;
    }
    .section-header {
        color: #8b5cf6;
        font-size: 1.08rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(139,92,246,0.35);
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.8em 1.8em;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        max-width: 260px;
        margin: auto;
        transition: 0.25s ease;
        box-shadow: 0 8px 20px rgba(59,130,246,0.25);
    }
    .stButton > button:hover { opacity: 0.92; transform: translateY(-1px); }
    .block-container { max-width: 1180px; margin: auto; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
EURO_TO_INR = 90.0
EURO_TO_USD = 1.08

# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI Connection
# ─────────────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/api"
ASSET_DIR = Path(__file__).resolve().parent / "assets"
train_metrics = {}
model_name = "FastAPI prediction service"

# control whether the brand quick-select expander is open
# remove quick-expander control (restoring original layout)


def get_brand_image_path(brand: str) -> Optional[Path]:
    mapping = {
        "dell": "dell.png",
        "hp": "hp.png",
        "lenovo": "lenovo.png",
        "apple": "apple.png",
        "asus": "asus.png",
        "msi": "msi.png",
        "samsung": "samsung.png",
        "razer": "razer.png",
        "microsoft": "Microsoft.png",
        "lg": "LG.png",
        "xiaomi": "xiaomi.png",
        "toshiba": "toshiba.png",
        "huawei": "huawei.png",
    }
    file_name = mapping.get(brand.lower())
    if not file_name:
        return None
    image_path = ASSET_DIR / "brands" / file_name
    return image_path if image_path.exists() else None

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
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">AI Price Estimator</div>
    <div class="hero-title">Laptop price prediction, made simple</div>
    <div class="hero-subtitle">
        Choose your laptop specs and get a smart estimate instantly with a cleaner, more modern experience.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

brand_names = ["Dell", "HP", "Lenovo", "Apple", "Asus", "MSI", "Samsung", "Razer", "Microsoft", "LG", "Xiaomi", "Toshiba", "Huawei"]
featured_brands = brand_names[:10]

st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<p class="section-header">Supported Brands</p>', unsafe_allow_html=True)
for row in [featured_brands[:5], featured_brands[5:10]]:
    cols = st.columns(5, gap="small")
    for brand, col in zip(row, cols):
        with col:
            image_path = get_brand_image_path(brand)
            if image_path is not None:
                st.image(str(image_path), width=90)
            else:
                st.markdown(f"<div style='color:#f8fafc'>{brand}</div>", unsafe_allow_html=True)
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

st.markdown('<p class="section-header">🔹 Brand & Type</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    company_options = ["Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Toshiba",
         "Apple", "Samsung", "Razer", "Microsoft", "Xiaomi", "LG", "Huawei", "Other"]
    default_company = st.session_state.get('company', company_options[0])
    try:
        default_index = company_options.index(default_company) if default_company in company_options else 0
    except Exception:
        default_index = 0
    company = st.selectbox(
        "Brand / Company",
        company_options,
        index=default_index,
        key='company',
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
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([1, 1, 1])
with center_col:
    predict_btn = st.button("🔍  Predict Price")

if predict_btn:
    # Send the form data to FastAPI. FastAPI performs validation and prediction.
    payload = {
        "company": company,
        "type_name": type_name,
        "os": os,
        "ram": int(ram),
        "cpu_brand": cpu_brand,
        "gpu_brand": gpu_brand,
        "weight": float(weight),
        "hdd": int(hdd),
        "ssd": int(ssd),
        "ppi": float(ppi),
        "touchscreen": int(touchscreen),
        "ips": int(ips),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()
        price_eur = float(prediction["predicted_price_euros"])
        model_name = prediction.get("model_name", model_name)
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure Uvicorn is running on port 8000.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The prediction API took too long to respond.")
        st.stop()
    except requests.exceptions.HTTPError as error:
        detail = response.json().get("detail", "Prediction request failed")
        st.error(f"API error ({error.response.status_code}): {detail}")
        st.stop()

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
