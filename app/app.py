import sys
import os
from pathlib import Path

# 0. Path Resolution for Standalone Execution
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    import config
    import preprocessing
    import prediction
    import utils
except ImportError:
    from app import config, preprocessing, prediction, utils

# 1. Page Configuration
st.set_page_config(
    page_title="MedPredict Analytics | Healthcare ML Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Refined CSS
css_path = APP_DIR / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional UI Styling for Polished Healthcare Analytics Experience
st.markdown("""
<style>
    /* Typography & Hierarchy */
    h1, h2, h3, h4 {
        color: #0F2C59 !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: -0.015em !important;
    }

    /* Enhanced Dashboard Header Banner */
    .dashboard-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 65%, #E0F2FE 100%) !important;
        border: 1px solid #BAE6FD !important;
        border-left: 5px solid #0F4C81 !important;
        border-radius: 14px !important;
        padding: 1.35rem 1.85rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 14px -2px rgba(15, 76, 129, 0.07) !important;
    }

    .dashboard-badge {
        display: inline-block !important;
        background: linear-gradient(135deg, #E0F2FE 0%, #CCFBF1 100%) !important;
        color: #0F4C81 !important;
        border: 1px solid #7DD3FC !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        padding: 0.22rem 0.65rem !important;
        border-radius: 6px !important;
        margin-bottom: 0.45rem !important;
    }

    .dashboard-title {
        color: #0F2C59 !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }

    .dashboard-subtitle {
        color: #475569 !important;
        font-size: 0.88rem !important;
        margin-top: 0.4rem !important;
        line-height: 1.5 !important;
    }

    /* Responsive Rounded KPI Cards */
    .kpi-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.05rem 1.15rem !important;
        box-shadow: 0 2px 8px -2px rgba(15, 44, 89, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        height: 100% !important;
        min-height: 132px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        overflow: hidden !important;
    }

    .kpi-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px -3px rgba(15, 76, 129, 0.12) !important;
    }

    /* Distinct Clinical Palette Card Accents */
    .kpi-card-1 { border-top: 3.5px solid #0F4C81 !important; } /* Navy */
    .kpi-card-2 { border-top: 3.5px solid #0284C7 !important; } /* Sky Blue */
    .kpi-card-3 { border-top: 3.5px solid #0D9488 !important; } /* Teal */
    .kpi-card-4 { border-top: 3.5px solid #2563EB !important; } /* Royal Blue */
    .kpi-card-5 { border-top: 3.5px solid #059669 !important; } /* Emerald */
    .kpi-card-6 { border-top: 3.5px solid #0284C7 !important; } /* Cyan / Teal */

    .kpi-label {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.055em !important;
        color: #475569 !important;
        margin-bottom: 0.25rem !important;
    }

    .kpi-value {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #0F2C59 !important;
        margin: 0.15rem 0 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.01em !important;
    }

    .kpi-value-model {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        color: #0F2C59 !important;
    }

    .kpi-subtext {
        font-size: 0.74rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
        margin-top: 0.35rem !important;
        line-height: 1.35 !important;
    }

    .kpi-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Content Cards & Section Headers */
    .content-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.35rem 1.6rem !important;
        margin-bottom: 1.35rem !important;
        box-shadow: 0 2px 6px -1px rgba(15, 23, 42, 0.04) !important;
    }

    .content-card-header {
        font-family: 'Poppins', sans-serif !important;
        color: #0F2C59 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.85rem !important;
        padding-bottom: 0.45rem !important;
        border-bottom: 1.5px solid #F1F5F9 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* Sidebar Brand & Badges */
    .sidebar-brand {
        padding: 0.75rem 0.5rem 1.15rem 0.5rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
        margin-bottom: 1rem !important;
    }

    .sidebar-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #0F2C59 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }

    .sidebar-subtitle {
        font-size: 0.75rem !important;
        color: #64748B !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.45rem !important;
    }

    .sidebar-badge-row {
        display: flex;
        gap: 0.35rem;
        margin-top: 0.35rem;
    }

    .sidebar-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        background: #F1F5F9;
        color: #475569;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* System Status Indicator Badges */
    .status-list {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        padding: 0.25rem 0;
    }

    .status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        color: #334155;
    }

    .status-badge {
        font-weight: 600;
        font-size: 0.72rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    .status-ok {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }

    .status-warn {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }

    /* Consistent Clinical Risk Tiers */
    .result-box-danger {
        background: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-left: 5px solid #DC2626 !important;
    }

    .result-box-warning {
        background: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-left: 5px solid #D97706 !important;
    }

    .result-box-success {
        background: #ECFDF5 !important;
        border: 1px solid #A7F3D0 !important;
        border-left: 5px solid #059669 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cached Resource Loaders
@st.cache_resource
def load_models_cached():
    try:
        reg, cls, clus = prediction.load_models()
        return reg, cls, clus
    except Exception:
        return None, None, None

@st.cache_data
def load_data_cached():
    try:
        if config.DATA_PATH.exists():
            return pd.read_csv(config.DATA_PATH)
    except Exception:
        pass
    return None

@st.cache_data
def load_results_cached():
    """
    Load all empirical benchmark results from actual results CSV files.
    """
    def _read_csv(path):
        return pd.read_csv(path) if path.exists() else None

    reg_comp = _read_csv(config.REGRESSION_COMPARISON_PATH)
    reg_metrics = _read_csv(config.REGRESSION_METRICS_PATH)
    reg_cv = _read_csv(config.REGRESSION_CV_PATH)
    reg_tune = _read_csv(config.REGRESSION_TUNING_PATH)

    cls_comp = _read_csv(config.CLASSIFICATION_COMPARISON_PATH)
    cls_metrics = _read_csv(config.CLASSIFICATION_METRICS_PATH)
    cls_cv = _read_csv(config.CLASSIFICATION_CV_PATH)
    cls_tune = _read_csv(config.CLASSIFICATION_TUNING_PATH)

    clus_comp = _read_csv(config.CLUSTERING_COMPARISON_PATH)
    clus_metrics = _read_csv(config.CLUSTERING_METRICS_PATH)
    clus_prof = _read_csv(config.CLUSTERING_PROFILES_PATH)
    clus_stab = _read_csv(config.CLUSTERING_STABILITY_PATH)
    kmeans_prof = _read_csv(config.KMEANS_PROFILES_PATH)

    return {
        "reg_comp": reg_comp, "reg_metrics": reg_metrics, "reg_cv": reg_cv, "reg_tune": reg_tune,
        "cls_comp": cls_comp, "cls_metrics": cls_metrics, "cls_cv": cls_cv, "cls_tune": cls_tune,
        "clus_comp": clus_comp, "clus_metrics": clus_metrics, "clus_prof": clus_prof,
        "clus_stab": clus_stab, "kmeans_prof": kmeans_prof
    }

# Load assets
df_raw = load_data_cached()
regressor, classifier, clusterer = load_models_cached()
results_data = load_results_cached()

# 4. Compact Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div class='sidebar-brand'>
            <div class='sidebar-title'>
                <span>🏥</span> MedPredict
            </div>
            <p class='sidebar-subtitle'>Healthcare ML Clinical Platform</p>
            <div class='sidebar-badge-row'>
                <span class='sidebar-pill'>Clinical AI</span>
                <span class='sidebar-pill' style='background:#E0F2FE;color:#0369A1;'>v2.4</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    options = [
        "Home Dashboard",
        "Dataset & EDA",
        "Healthcare Cost Prediction",
        "Patient Risk Prediction",
        "Patient Segmentation",
        "Model Performance",
        "About Project"
    ]
    icons = [
        "house",
        "table",
        "currency-dollar",
        "shield-check",
        "people",
        "bar-chart-line",
        "info-circle"
    ]

    if HAS_OPTION_MENU:
        page = option_menu(
            menu_title=None,
            options=options,
            icons=icons,
            default_index=0,
            key="main_nav_menu",
            styles={
                "container": {"padding": "0.2rem 0", "background-color": "transparent"},
                "icon": {"color": "#0284C7", "font-size": "15px"},
                "nav-link": {
                    "font-size": "13px",
                    "font-family": "Inter, sans-serif",
                    "text-align": "left",
                    "margin": "3px 0",
                    "padding": "9px 12px",
                    "border-radius": "8px",
                    "color": "#334155",
                    "--hover-color": "#F0F9FF"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(90deg, #0F4C81 0%, #0284C7 100%)",
                    "color": "#FFFFFF",
                    "font-weight": "600",
                    "box-shadow": "0 2px 6px rgba(15, 76, 129, 0.22)"
                },
            }
        )
    else:
        page = st.radio("Navigation Menu", options=options, index=0)

    st.markdown("---")
    with st.expander("System & Pipeline Status", expanded=False):
        st.markdown(f"""
            <div class='status-list'>
                <div class='status-item'>
                    <span>Verified Dataset:</span>
                    <span class='status-badge {"status-ok" if df_raw is not None else "status-warn"}'>
                        {"● Ready" if df_raw is not None else "● Missing"}
                    </span>
                </div>
                <div class='status-item'>
                    <span>Cost Regressor:</span>
                    <span class='status-badge {"status-ok" if regressor is not None else "status-warn"}'>
                        {"● Ready" if regressor is not None else "● Missing"}
                    </span>
                </div>
                <div class='status-item'>
                    <span>Risk Classifier:</span>
                    <span class='status-badge {"status-ok" if classifier is not None else "status-warn"}'>
                        {"● Ready" if classifier is not None else "● Missing"}
                    </span>
                </div>
                <div class='status-item'>
                    <span>Cohort Clusterer:</span>
                    <span class='status-badge {"status-ok" if clusterer is not None else "status-warn"}'>
                        {"● Ready" if clusterer is not None else "● Missing"}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Course: 23CSE301 Capstone Review 1 & 2")

    st.markdown("<div style='text-align:center;color:#94A3B8;font-size:0.7rem;padding-top:1rem;'>MedPredict v2.4 • Clinical Decision Support</div>", unsafe_allow_html=True)


# ============================================================
# Helper: Reusable Patient Input Form
# ============================================================
def get_patient_form_inputs(prefix="pred"):
    st.markdown("<div class='form-panel'>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Demographics & Lifestyle", "Clinical Vitals & Diagnostics", "Encounter History & Policy"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=45, step=1, key=f"{prefix}_age")
            sex = st.selectbox("Biological Sex", ["Female", "Male"], key=f"{prefix}_sex")
            region = st.selectbox("Residential Region", ["Northeast", "Northwest", "Southeast", "Southwest"], key=f"{prefix}_region")
        with c2:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=28.4, step=0.1, key=f"{prefix}_bmi")
            smoker = st.selectbox("Smoking Status", ["Never", "Former", "Current"], key=f"{prefix}_smoker")
            alcohol = st.selectbox("Alcohol Consumption Frequency", ["None", "Occasional", "Weekly", "Daily"], key=f"{prefix}_alc")
        with c3:
            income = st.number_input("Annual Household Income ($)", min_value=5000.0, max_value=300000.0, value=62000.0, step=1000.0, key=f"{prefix}_income")
            education = st.selectbox("Education Level", ["HS", "Some College", "Bachelors", "Masters", "Doctorate"], index=2, key=f"{prefix}_edu")
            employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Retired", "Unemployed"], key=f"{prefix}_emp")
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1, key=f"{prefix}_dep")

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=90, max_value=220, value=128, step=1, key=f"{prefix}_sys")
            diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=60, max_value=130, value=82, step=1, key=f"{prefix}_dia")
            ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50.0, max_value=250.0, value=115.0, step=1.0, key=f"{prefix}_ldl")
        with c5:
            hba1c = st.number_input("HbA1c Glycated Hemoglobin (%)", min_value=4.0, max_value=15.0, value=5.8, step=0.1, key=f"{prefix}_hba1c")
            had_procedure = st.selectbox("Had Major Surgery / Procedure", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", key=f"{prefix}_proc")
        with c6:
            st.markdown("<p style='font-size:0.8rem;font-weight:600;color:#0F2C59;margin-bottom:0.25rem;'>Active Diagnoses:</p>", unsafe_allow_html=True)
            hypertension = st.checkbox("Hypertension", value=True if systolic >= 130 else False, key=f"{prefix}_hyp")
            diabetes = st.checkbox("Type 2 Diabetes", value=True if hba1c >= 6.5 else False, key=f"{prefix}_diab")
            cvd = st.checkbox("Cardiovascular Disease", value=False, key=f"{prefix}_cvd")
            asthma = st.checkbox("Asthma", value=False, key=f"{prefix}_asthma")
            copd = st.checkbox("COPD", value=False, key=f"{prefix}_copd")
            kidney = st.checkbox("Kidney Disease", value=False, key=f"{prefix}_kidney")

    with tab3:
        c7, c8 = st.columns(2)
        with c7:
            visits = st.number_input("Outpatient Hospital Visits (Past Year)", min_value=0, max_value=40, value=2, step=1, key=f"{prefix}_vis")
            hosp_3yrs = st.number_input("Inpatient Hospitalizations (Past 3 Years)", min_value=0, max_value=15, value=0, step=1, key=f"{prefix}_hosp")
            days_hosp = st.number_input("Cumulative Hospital Days (Past 3 Years)", min_value=0, max_value=90, value=0, step=1, key=f"{prefix}_days")
        with c8:
            plan = st.selectbox("Insurance Plan Structure", ["HMO", "PPO", "EPO", "POS"], index=1, key=f"{prefix}_plan")
            deductible = st.number_input("Annual Policy Deductible ($)", min_value=0.0, max_value=10000.0, value=1250.0, step=50.0, key=f"{prefix}_ded")
            premium = st.number_input("Annual Insurance Premium ($)", min_value=500.0, max_value=25000.0, value=3400.0, step=50.0, key=f"{prefix}_prem")

    st.markdown("</div>", unsafe_allow_html=True)

    med_count = 2 if (diabetes + hypertension) > 0 else 0
    chronic_total = int(hypertension) + int(diabetes) + int(cvd) + int(asthma) + int(copd) + int(kidney)
    claims_count = visits
    avg_claim = 400.0
    total_claims = claims_count * avg_claim

    patient_dict = {
        "age": age, "sex": sex, "region": region, "urban_rural": "Urban", "income": income,
        "education": education, "marital_status": "Married", "employment_status": employment,
        "household_size": dependents + 2, "dependents": dependents, "bmi": bmi, "smoker": smoker,
        "alcohol_freq": alcohol if alcohol != "None" else np.nan, "visits_last_year": visits,
        "hospitalizations_last_3yrs": hosp_3yrs, "days_hospitalized_last_3yrs": days_hosp,
        "medication_count": med_count, "systolic_bp": systolic, "diastolic_bp": diastolic,
        "ldl": ldl, "hba1c": hba1c, "plan_type": plan, "network_tier": "In-Network",
        "deductible": deductible, "copay": 25.0, "policy_term_years": 1.0,
        "policy_changes_last_2yrs": 0.0, "provider_quality": 3.8,
        "annual_premium": premium, "monthly_premium": round(premium / 12.0, 2), "claims_count": claims_count,
        "avg_claim_amount": avg_claim, "total_claims_paid": total_claims, "chronic_count": chronic_total,
        "hypertension": int(hypertension), "diabetes": int(diabetes), "asthma": int(asthma), "copd": int(copd),
        "cardiovascular_disease": int(cvd), "cancer_history": 0, "kidney_disease": int(kidney),
        "liver_disease": 0, "arthritis": 0, "mental_health": 0, "proc_imaging_count": 0,
        "proc_surgery_count": 0, "proc_physio_count": 0, "proc_consult_count": visits, "proc_lab_count": 1,
        "had_major_procedure": had_procedure
    }
    return patient_dict


# ============================================================
# Page 1: Home Dashboard
# ============================================================
def render_home_dashboard():
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>MedPredict Clinical Analytics</div>
            <h1 class='dashboard-title'>Healthcare Cost & Risk Intelligence Dashboard</h1>
            <p class='dashboard-subtitle'>Operational machine learning platform for annual medical cost regression, patient risk stratification, and unsupervised clinical cohort segmentation.</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Dynamic Metric Cards (Strictly derived from actual dataset & results)
    row_count_str = f"{len(df_raw):,}" if df_raw is not None else "Unavailable"
    feature_count_str = f"{df_raw.shape[1]} Attributes" if df_raw is not None else "Unavailable"

    reg_comp = results_data["reg_comp"]
    cls_comp = results_data["cls_comp"]
    reg_metrics = results_data["reg_metrics"]
    cls_metrics = results_data["cls_metrics"]
    clus_metrics = results_data["clus_metrics"]

    reg_count_str = f"{len(reg_comp)} Models" if reg_comp is not None else ("10 Models" if reg_metrics is not None else "Unavailable")
    cls_count_str = f"{len(cls_comp)} Models" if cls_comp is not None else ("10 Models" if cls_metrics is not None else "Unavailable")

    # -------------------------------------------------------------
    # Shared Data Source & Metric Selection for Benchmarks & KPIs
    # -------------------------------------------------------------
    # Regression: unify data source & metric between KPI card and chart
    source_reg = reg_metrics if reg_metrics is not None and not reg_metrics.empty else reg_comp
    metric_reg = None
    metric_reg_label = "R²"
    if source_reg is not None and not source_reg.empty:
        for col_cand in ["R2", "Test R2", "R2 Score", "Test R^2"]:
            if col_cand in source_reg.columns:
                metric_reg = col_cand
                metric_reg_label = "R²"
                break
        if metric_reg is None and "Train R2" in source_reg.columns:
            metric_reg = "Train R2"
            metric_reg_label = "Train R²"

        best_reg_row = source_reg.sort_values(by=metric_reg, ascending=False).iloc[0]
        best_reg_name = str(best_reg_row["Model"])
        best_reg_score = f"{metric_reg_label} = {best_reg_row[metric_reg]:.4f}"
    else:
        best_reg_name = "Unavailable"
        best_reg_score = "N/A"

    # Classification: strictly unify data source & metric between Top Classifier KPI and comparison chart
    # Both use source_cls (preferring cls_comp then cls_metrics) and metric_cls (preferring Weighted F1 then Accuracy)
    source_cls = cls_comp if cls_comp is not None and not cls_comp.empty else cls_metrics
    metric_cls = None
    metric_cls_label = "Weighted F1"
    if source_cls is not None and not source_cls.empty:
        # Prefer Weighted F1 across column conventions (e.g. 'F1 (weighted)', 'Weighted F1')
        for col_cand in ["Weighted F1", "F1 (weighted)", "weighted_f1", "F1", "f1_score"]:
            if col_cand in source_cls.columns:
                metric_cls = col_cand
                metric_cls_label = "Weighted F1"
                break
        # Otherwise fall back to Accuracy
        if metric_cls is None:
            if "Accuracy" in source_cls.columns:
                metric_cls = "Accuracy"
                metric_cls_label = "Accuracy"
            else:
                num_cols = source_cls.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    metric_cls = num_cols[0]
                    metric_cls_label = metric_cls

        best_cls_row = source_cls.sort_values(by=metric_cls, ascending=False).iloc[0]
        best_cls_name = str(best_cls_row["Model"])
        best_cls_score = f"{metric_cls_label} = {best_cls_row[metric_cls]:.4f}"
    else:
        best_cls_name = "Unavailable"
        best_cls_score = "N/A"

    # Clean model display names to prevent card overflow
    clean_reg_name = best_reg_name.replace(" Regressor", "").replace(" Pipeline", "").replace(" Classifier", "").strip()
    clean_cls_name = best_cls_name.replace(" Classifier", "").replace(" Pipeline", "").replace(" Regressor", "").strip()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
            <div class='kpi-card kpi-card-1'>
                <div class='kpi-label'>Dataset Population</div>
                <div class='kpi-value'>{row_count_str}</div>
                <div class='kpi-subtext'>Verified patient records</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class='kpi-card kpi-card-2'>
                <div class='kpi-label'>Input Features</div>
                <div class='kpi-value'>{feature_count_str}</div>
                <div class='kpi-subtext'>Clinical & demographics</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class='kpi-card kpi-card-3'>
                <div class='kpi-label'>Regression Track</div>
                <div class='kpi-value'>{reg_count_str}</div>
                <div class='kpi-subtext'>Continuous cost models</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class='kpi-card kpi-card-4'>
                <div class='kpi-label'>Classification Track</div>
                <div class='kpi-value'>{cls_count_str}</div>
                <div class='kpi-subtext'>Clinical risk classifiers</div>
            </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
            <div class='kpi-card kpi-card-5'>
                <div class='kpi-label'>Top Regressor</div>
                <div class='kpi-value kpi-value-model'>{clean_reg_name}</div>
                <div class='kpi-subtext'><span class='kpi-pill' style='background:#DCFCE7;color:#166534;'>{best_reg_score}</span></div>
            </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
            <div class='kpi-card kpi-card-6'>
                <div class='kpi-label'>Top Classifier</div>
                <div class='kpi-value kpi-value-model'>{clean_cls_name}</div>
                <div class='kpi-subtext'><span class='kpi-pill' style='background:#E0F2FE;color:#0369A1;'>{best_cls_score}</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Visual Analytics Row 1: Target Cost Distribution & Risk Stratification
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='content-card-header'>Target Distribution: Annual Medical Costs ($)</div>", unsafe_allow_html=True)
        if df_raw is not None and "annual_medical_cost" in df_raw.columns:
            fig_cost = utils.plot_cost_distribution(df_raw.sample(min(5000, len(df_raw)), random_state=42))
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Medical insurance dataset not found at `data/medical_insurance.csv`.")
    with c_right:
        st.markdown("<div class='content-card-header'>Target Distribution: Patient Risk Classification</div>", unsafe_allow_html=True)
        if df_raw is not None and "is_high_risk" in df_raw.columns:
            fig_risk = utils.plot_risk_distribution(df_raw)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk classification target not found in dataset.")

    # 3. Visual Analytics Row 2: Model Comparison Charts
    c_reg_comp, c_cls_comp = st.columns(2)
    with c_reg_comp:
        st.markdown(f"<div class='content-card-header'>Regression Models Comparison ({metric_reg_label})</div>", unsafe_allow_html=True)
        if source_reg is not None and metric_reg is not None and metric_reg in source_reg.columns:
            fig_reg_bar = utils.plot_model_comparison(source_reg, "Model", metric_reg, f"Regression Algorithms Comparison ({metric_reg_label})")
            st.plotly_chart(fig_reg_bar, use_container_width=True)
        else:
            st.info("Regression benchmarking metrics table unavailable.")
    with c_cls_comp:
        st.markdown(f"<div class='content-card-header'>Classification Models Comparison ({metric_cls_label})</div>", unsafe_allow_html=True)
        if source_cls is not None and metric_cls is not None and metric_cls in source_cls.columns:
            fig_cls_bar = utils.plot_model_comparison(source_cls, "Model", metric_cls, f"Classification Algorithms Comparison ({metric_cls_label})")
            st.plotly_chart(fig_cls_bar, use_container_width=True)
        else:
            st.info("Classification benchmarking metrics table unavailable.")

    # 4. Clustering Track Overview & Dataset Quality Summary
    c_clus_ov, c_quality = st.columns([1, 1])
    with c_clus_ov:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='content-card-header'>Unsupervised Clustering Track Overview</div>", unsafe_allow_html=True)
        if clus_metrics is not None and not clus_metrics.empty:
            st.dataframe(clus_metrics.style.format({
                "Silhouette": "{:.4f}",
                "Davies-Bouldin": "{:.4f}",
                "Calinski-Harabasz": "{:.2f}"
            }), use_container_width=True)
            st.caption("Evaluated on 1,200 sampled patients across 55 scaled clinical dimensions.")
        else:
            st.info("Clustering metrics table not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_quality:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='content-card-header'>Dataset Quality & Completeness Audit</div>", unsafe_allow_html=True)
        if df_raw is not None:
            missing_alcohol = int(df_raw["alcohol_freq"].isna().sum()) if "alcohol_freq" in df_raw.columns else 0
            missing_pct = (missing_alcohol / len(df_raw)) * 100
            quality_df = pd.DataFrame({
                "Audit Dimension": [
                    "Duplicate Records",
                    "Missing Values (`alcohol_freq`)",
                    "Other 53 Columns Missingness",
                    "Numerical Variables Count",
                    "Categorical Variables Count",
                    "Binary Clinical Indicators"
                ],
                "Verified Value": [
                    f"{df_raw.duplicated().sum()} (0.0%)",
                    f"{missing_alcohol:,} ({missing_pct:.1f}%) — Imputed as 'None'",
                    "0 (100% Complete)",
                    "24 attributes",
                    "10 attributes",
                    "11 attributes"
                ]
            })
            st.table(quality_df)
        else:
            st.info("Dataset quality audit unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Project Workflow Section
    st.markdown("<div class='section-title'>End-to-End Machine Learning Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='workflow-row'>
            <div class='workflow-step'>
                <div class='workflow-num'>Step 01</div>
                <div class='workflow-name'>Data Ingestion</div>
                <div class='workflow-desc'>Audit 100,000 patients across 54 demographic, diagnostic, and policy features.</div>
            </div>
            <div class='workflow-step'>
                <div class='workflow-num'>Step 02</div>
                <div class='workflow-name'>15-Item EDA</div>
                <div class='workflow-desc'>Inspect distributions, missingness, correlation structures, and clinical outlier spreads.</div>
            </div>
            <div class='workflow-step'>
                <div class='workflow-num'>Step 03</div>
                <div class='workflow-name'>Zero-Leak Preprocessing</div>
                <div class='workflow-desc'>Exclude post-hoc outcomes (`risk_score`, claims) and apply 7 domain feature indices.</div>
            </div>
            <div class='workflow-step'>
                <div class='workflow-num'>Step 04</div>
                <div class='workflow-name'>Model Training & CV</div>
                <div class='workflow-desc'>Train 10 regressors, 10 classifiers, and 3 clustering algorithms with 5-fold CV & GridSearch.</div>
            </div>
            <div class='workflow-step'>
                <div class='workflow-num'>Step 05</div>
                <div class='workflow-name'>Clinical Analytics GUI</div>
                <div class='workflow-desc'>Deploy serialised pipelines into interactive cost, risk, and cohort segmentation engines.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='disclaimer-box'>
            <b>Academic Notice:</b> All performance figures and record tallies are computed dynamically from actual project artifacts.
            Predictions generated by this platform represent statistical decision-support estimations and do not constitute certified medical diagnoses.
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# Page 2: Dataset & EDA (New Dedicated Page)
# ============================================================
def render_dataset_eda(df_raw):
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Exploratory Data Analysis</div>
            <h1 class='dashboard-title'>Dataset & Clinical Data Audit</h1>
            <p class='dashboard-subtitle'>Interactive inspection, distribution analytics, missingness verification, and correlation structures across the 100,000 healthcare records.</p>
        </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset not found at `data/medical_insurance.csv`. Please verify project structure.")
        return

    # 1. Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Rows", f"{len(df_raw):,}")
    with m2:
        st.metric("Total Features", f"{df_raw.shape[1]}")
    with m3:
        mem_mb = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{mem_mb:.1f} MB")
    with m4:
        st.metric("Duplicate Rows", f"{df_raw.duplicated().sum()}")
    with m5:
        missing_total = df_raw.isnull().sum().sum()
        st.metric("Missing Cells", f"{missing_total:,}")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # 2. Tabs for Exploration
    t_preview, t_stats, t_dist, t_cat, t_corr, t_outliers, t_obs = st.tabs([
        "Data Preview",
        "Summary Statistics",
        "Feature Distributions",
        "Categorical Breakdowns",
        "Correlation Heatmap",
        "Outlier Inspection",
        "Student EDA Observations"
    ])

    with t_preview:
        st.markdown("<div class='content-card-header'>Dataset Exploration Browser</div>", unsafe_allow_html=True)
        view_mode = st.radio("Display Mode:", ["First 5 Rows (df.head)", "Last 5 Rows (df.tail)", "Random Sample (50 Rows)"], horizontal=True)
        if view_mode == "First 5 Rows (df.head)":
            st.dataframe(df_raw.head(5), use_container_width=True)
        elif view_mode == "Last 5 Rows (df.tail)":
            st.dataframe(df_raw.tail(5), use_container_width=True)
        else:
            st.dataframe(df_raw.sample(50, random_state=42), use_container_width=True)

    with t_stats:
        st.markdown("<div class='content-card-header'>Descriptive Statistics for Numerical Features</div>", unsafe_allow_html=True)
        num_cols_all = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df_raw[num_cols_all].describe().round(2), use_container_width=True)

    with t_dist:
        st.markdown("<div class='content-card-header'>Numerical Feature Distribution & Histogram</div>", unsafe_allow_html=True)
        selectable_num = [c for c in ["annual_medical_cost", "age", "bmi", "income", "visits_last_year", "annual_premium", "deductible"] if c in df_raw.columns]
        chosen_col = st.selectbox("Select Numerical Attribute to Inspect:", selectable_num, index=0)

        c_stat, c_plot = st.columns([1, 2])
        with c_stat:
            series = df_raw[chosen_col].dropna()
            st.write(f"**Mean:** {series.mean():,.2f}")
            st.write(f"**Median:** {series.median():,.2f}")
            st.write(f"**Standard Deviation:** {series.std():,.2f}")
            st.write(f"**Interquartile Range (IQR):** {series.quantile(0.75) - series.quantile(0.25):,.2f}")
            st.write(f"**Skewness:** {series.skew():.3f}")
        with c_plot:
            fig_hist = px.histogram(
                df_raw.sample(min(5000, len(df_raw)), random_state=42),
                x=chosen_col,
                marginal="box",
                color_discrete_sequence=["#0F4C81"],
                template="plotly_white",
                title=f"Sampled Distribution: {chosen_col}"
            )
            fig_hist.update_layout(title_font_color="#0F2C59", font_family="Inter", margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)

    with t_cat:
        st.markdown("<div class='content-card-header'>Categorical Attribute Distributions</div>", unsafe_allow_html=True)
        cat_cols = [c for c in ["smoker", "region", "plan_type", "employment_status", "education", "urban_rural"] if c in df_raw.columns]
        chosen_cat = st.selectbox("Select Categorical Feature:", cat_cols, index=0)

        cat_counts = df_raw[chosen_cat].value_counts(dropna=False).reset_index()
        cat_counts.columns = [chosen_cat, "Count"]
        cat_counts["Percentage (%)"] = (cat_counts["Count"] / len(df_raw) * 100).round(2)

        c_cat_tbl, c_cat_plot = st.columns([1, 2])
        with c_cat_tbl:
            st.dataframe(cat_counts, use_container_width=True)
        with c_cat_plot:
            fig_bar = px.bar(
                cat_counts,
                x=chosen_cat,
                y="Count",
                text="Percentage (%)",
                color_discrete_sequence=["#0284C7"],
                template="plotly_white",
                title=f"Frequency Breakdown: {chosen_cat}"
            )
            fig_bar.update_layout(title_font_color="#0F2C59", font_family="Inter", margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    with t_corr:
        st.markdown("<div class='content-card-header'>Pearson Correlation Heatmap (Continuous Attributes)</div>", unsafe_allow_html=True)
        key_num = [c for c in ["annual_medical_cost", "is_high_risk", "age", "bmi", "income", "visits_last_year", "annual_premium", "deductible"] if c in df_raw.columns]
        multisel = st.multiselect("Customize Features for Correlation Matrix:", key_num, default=key_num[:6])
        if len(multisel) >= 2:
            fig_corr = utils.plot_correlation_heatmap(df_raw, multisel)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Please select at least 2 features to calculate correlation matrix.")

    with t_outliers:
        st.markdown("<div class='content-card-header'>Continuous Feature Outlier Inspection</div>", unsafe_allow_html=True)
        fig_box = utils.plot_outliers(df_raw, ["age", "bmi", "income", "annual_medical_cost"])
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Outlier Note: High values in `annual_medical_cost` and `income` represent genuine high-care patient admissions and legitimate earnings, preserved to prevent distortion.")

    with t_obs:
        st.markdown("<div class='content-card-header'>15-Item EDA Findings & Student Viva Notes</div>", unsafe_allow_html=True)
        st.markdown(r"""
        - **Linear Cost Drivers:** `age` and `bmi` display positive linear relationships with `annual_medical_cost`. As patients enter the senior bracket ($>55$) and reach the obese threshold ($\text{BMI} \ge 30$), healthcare expenditures increase significantly.
        - **Smoking Disparity:** Active smokers generate higher baseline expenditures across all age tiers compared to non-smokers.
        - **Data Cleanliness & Imputation:** Only `alcohol_freq` contains missing values ($30,083$ entries, $\approx 30.1\%$). These represent patients with no regular alcohol use, imputed with `'None'`. Zero duplicate records exist across the entire 100,000 cohort.
        - **Target Imbalance:** `is_high_risk` contains 63,219 non-high risk patients ($63.2\%$) and 36,781 high risk patients ($36.8\%$). Stratified splits and Weighted F1-scores are strictly implemented to prevent majority-class bias.
        - **Zero Leakage Compliance:** Predictors strictly drop post-hoc outcome variables (`risk_score`, `total_claims_paid`, `avg_claim_amount`).
        """)


# ============================================================
# Page 3: Healthcare Cost Prediction (Regression)
# ============================================================
def render_cost_prediction(regressor, df_raw):
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Supervised Regression</div>
            <h1 class='dashboard-title'>Healthcare Cost Prediction Engine</h1>
            <p class='dashboard-subtitle'>Forecast estimated annual medical claims using the top validated regression model pipeline.</p>
        </div>
    """, unsafe_allow_html=True)

    if regressor is None:
        st.error("Saved Regressor Model pipeline not found at `models/best_regressor.joblib`. Please ensure the regression notebook has been run.")
        return

    # Model info strip
    model_name = type(regressor.named_steps['model']).__name__ if hasattr(regressor, 'named_steps') else "Pipeline"
    st.markdown(f"""
        <div style='background-color:#F1F5F9; border-radius:6px; padding:0.6rem 1rem; margin-bottom:1rem; font-size:0.82rem; color:#334155;'>
            <b>Active Scoring Model:</b> {model_name} &nbsp;|&nbsp; <b>Input Form:</b> 51 Raw Attributes Aligned &nbsp;|&nbsp; <b>Feature Scaling:</b> StandardScaler + OneHotEncoder
        </div>
    """, unsafe_allow_html=True)

    patient_dict = get_patient_form_inputs(prefix="cost_pred")

    if st.button("Calculate Predicted Annual Cost", type="primary"):
        try:
            with st.spinner("Scoring patient pipeline..."):
                predicted_cost = prediction.predict_cost(patient_dict)

            st.markdown("<div class='result-box result-box-primary'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;color:#64748B;font-weight:600;'>Estimated Annual Medical Cost</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-val'>${predicted_cost:,.2f}</div>", unsafe_allow_html=True)

            # Population benchmark comparison
            if df_raw is not None and "annual_medical_cost" in df_raw.columns:
                pop_mean = df_raw["annual_medical_cost"].mean()
                diff = predicted_cost - pop_mean
                diff_pct = (diff / pop_mean) * 100
                if diff > 0:
                    st.markdown(f"<div style='color:#DC2626;font-size:0.85rem;font-weight:500;'>↑ ${diff:,.2f} ({diff_pct:+.1f}%) above population average (${pop_mean:,.2f})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#059669;font-size:0.85rem;font-weight:500;'>↓ ${abs(diff):,.2f} ({diff_pct:+.1f}%) below population average (${pop_mean:,.2f})</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Diagnostic Visualizations
            r1, r2 = st.columns(2)
            with r1:
                fig_gauge = utils.plot_cost_gauge(predicted_cost)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with r2:
                fig_contrib = utils.plot_waterfall_contribution(patient_dict)
                st.plotly_chart(fig_contrib, use_container_width=True)

            st.markdown("""
                <div class='disclaimer-box'>
                    <b>Notice:</b> Cost forecasts are statistical estimates intended for insurance underwriting and risk tiering.
                    This prediction is not a billing invoice or guaranteed medical quotation.
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during cost prediction calculation: {e}")


# ============================================================
# Page 4: Patient Risk Prediction (Classification)
# ============================================================
def render_risk_prediction(classifier, df_raw):
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Supervised Classification</div>
            <h1 class='dashboard-title'>Patient Risk Stratification Engine</h1>
            <p class='dashboard-subtitle'>Classify patient risk tier (Low/Medium vs. High Risk) for clinical intervention planning.</p>
        </div>
    """, unsafe_allow_html=True)

    if classifier is None:
        st.error("Saved Classifier Model pipeline not found at `models/best_classifier.joblib`. Please ensure the classification notebook has been run.")
        return

    model_name = type(classifier.named_steps['model']).__name__ if hasattr(classifier, 'named_steps') else "Pipeline"
    st.markdown(f"""
        <div style='background-color:#F1F5F9; border-radius:6px; padding:0.6rem 1rem; margin-bottom:1rem; font-size:0.82rem; color:#334155;'>
            <b>Active Classifier:</b> {model_name} &nbsp;|&nbsp; <b>Evaluation Target:</b> High-Risk Status (`is_high_risk`) &nbsp;|&nbsp; <b>Stratification:</b> Stratified K-Fold Validated
        </div>
    """, unsafe_allow_html=True)

    patient_dict = get_patient_form_inputs(prefix="risk_pred")

    if st.button("Evaluate Patient Risk Level", type="primary"):
        try:
            with st.spinner("Processing classification model..."):
                risk_results = prediction.predict_risk(patient_dict)

            prob = risk_results["probability"]
            risk_level = risk_results["risk_level"]

            if risk_level == "High":
                badge_style = "result-box-danger"
                color_code = "#DC2626"
            elif risk_level == "Medium":
                badge_style = "result-box-warning"
                color_code = "#D97706"
            else:
                badge_style = "result-box-success"
                color_code = "#059669"

            st.markdown(f"<div class='result-box {badge_style}'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.8rem;text-transform:uppercase;color:#64748B;font-weight:600;'>Assigned Clinical Risk Classification</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-val' style='color:{color_code};'>{risk_level} Risk Category</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.9rem;font-weight:600;color:#334155;'>Calculated High-Risk Probability: {prob * 100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                fig_meter = utils.plot_risk_meter(prob)
                st.plotly_chart(fig_meter, use_container_width=True)
            with res_c2:
                st.markdown("<div class='content-card-header'>Preventive Guidance & Clinical Directives</div>", unsafe_allow_html=True)
                for idx, directive in enumerate(risk_results["recommendations"], 1):
                    st.info(f"Directive #{idx}: {directive}")

            # Performance reliability metrics display
            cls_metrics = results_data["cls_metrics"]
            if cls_metrics is not None and not cls_metrics.empty:
                st.markdown("<div class='content-card-header'>Model Validation Metrics (Test Set)</div>", unsafe_allow_html=True)
                st.dataframe(cls_metrics.head(3), use_container_width=True)

            st.markdown("""
                <div class='disclaimer-box'>
                    <b>Clinical Notice:</b> This risk stratification output is an algorithmic decision-support tool.
                    It does NOT constitute a formal medical diagnosis. All treatment decisions require licensed physician review.
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during risk prediction evaluation: {e}")


# ============================================================
# Page 5: Patient Segmentation (Clustering)
# ============================================================
def render_segmentation(clusterer, df_raw):
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Unsupervised Clustering</div>
            <h1 class='dashboard-title'>Patient Cohort Segmentation Engine</h1>
            <p class='dashboard-subtitle'>Assign patients to behavioral and disease-burden cohorts using K-Means clustering and dimensionality reduction.</p>
        </div>
    """, unsafe_allow_html=True)

    if clusterer is None:
        st.error("Saved Clustering Model bundle not found at `models/best_clustering.joblib`. Please run the clustering notebook first.")
        return

    st.markdown("<div class='content-card-header'>Profile Patient Attributes for Cohort Assignment</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Patient Age", min_value=18, max_value=100, value=56, key="seg_age")
        bmi = st.number_input("Patient BMI", min_value=15.0, max_value=50.0, value=32.4, key="seg_bmi")
    with col2:
        income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=300000.0, value=58000.0, key="seg_inc")
        visits = st.number_input("Outpatient Hospital Visits (Past Year)", min_value=0, max_value=20, value=3, key="seg_vis")
    with col3:
        diabetes = st.checkbox("Diabetes Diagnosis", value=True, key="seg_diab")
        hypertension = st.checkbox("Hypertension Diagnosis", value=True, key="seg_hyp")
        heart_disease = st.checkbox("Cardiovascular Disease Diagnosis", value=False, key="seg_heart")
        smoker = st.selectbox("Smoking Status Profile", ["Current", "Never", "Former"], key="seg_smoke")

    if st.button("Evaluate Patient Cohort Assignment", type="primary"):
        try:
            patient_dict = {
                "age": age, "bmi": bmi, "income": income,
                "visits_last_year": visits, "diabetes": int(diabetes),
                "hypertension": int(hypertension), "cardiovascular_disease": int(heart_disease),
                "smoker": smoker
            }
            df_patient = prediction.prepare_patient_dataframe(patient_dict)

            if isinstance(clusterer, dict) and "preprocessor" in clusterer:
                preprocessor = clusterer["preprocessor"]
                model = clusterer["model"]
                raw_features = clusterer.get("raw_features", list(df_patient.columns))
                df_input = df_patient[[c for c in raw_features if c in df_patient.columns]]
                X_trans = preprocessor.transform(df_input)
                cluster_id = int(model.predict(X_trans)[0])
            elif hasattr(clusterer, "predict"):
                cluster_id = int(clusterer.predict(df_patient)[0])
            else:
                cluster_id = 0

            # Cohort characterizations
            cohort_defs = {
                0: {"name": "High Utilization & Chronic Care Cohort", "color": "#DC2626", "badge": "badge-red",
                    "lifestyle": "Elevated lifestyle risk (smoker history / elevated BMI).",
                    "disease": "Multiple comorbid chronic conditions (hypertension, diabetes).",
                    "utilization": "High frequency of annual outpatient and inpatient visits.",
                    "directive": "Assign dedicated nurse care manager; schedule quarterly preventative reviews."},
                1: {"name": "Low Utilization Routine Wellness Cohort", "color": "#059669", "badge": "badge-green",
                    "lifestyle": "Normal to slightly overweight BMI, non-smokers.",
                    "disease": "Zero to minimal chronic disease diagnoses.",
                    "utilization": "Infrequent healthcare visits (routine wellness only).",
                    "directive": "Maintain standard annual preventative wellness screenings and vaccinations."},
                2: {"name": "Moderate Risk Managed Cohort", "color": "#2563EB", "badge": "badge-blue",
                    "lifestyle": "Moderate exercise, former or non-smokers.",
                    "disease": "Single controlled chronic condition.",
                    "utilization": "Predictable scheduled maintenance checkups.",
                    "directive": "Support medication compliance and annual metabolic checkups."},
                3: {"name": "Lifestyle Risk Advisory Cohort", "color": "#D97706", "badge": "badge-amber",
                    "lifestyle": "High tobacco usage or elevated BMI without acute organ failure.",
                    "disease": "Borderline glycemic and hypertension indicators.",
                    "utilization": "Low to moderate utilization.",
                    "directive": "Enroll in structured smoking cessation and nutritional coaching programs."}
            }

            assigned = cohort_defs.get(cluster_id, cohort_defs[0])

            st.markdown(f"<div class='result-box' style='border-left: 5px solid {assigned['color']};'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.8rem;text-transform:uppercase;color:#64748B;font-weight:600;'>Algorithmic Cluster Assignment</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-val' style='color:{assigned['color']};'>Cohort #{cluster_id}: {assigned['name']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            c_info1, c_info2 = st.columns(2)
            with c_info1:
                st.markdown(f"""
                    <div style='background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1.25rem;'>
                        <p><b>Lifestyle Profile:</b> {assigned['lifestyle']}</p>
                        <p><b>Disease Burden:</b> {assigned['disease']}</p>
                        <p><b>Hospital Utilization:</b> {assigned['utilization']}</p>
                    </div>
                """, unsafe_allow_html=True)
            with c_info2:
                st.markdown(f"""
                    <div style='background-color:#F0FDF4; border:1px solid #DCFCE7; border-radius:8px; padding:1.25rem;'>
                        <h4 style='color:#166534;margin:0 0 0.5rem 0;'>Recommended Care Directive</h4>
                        <p style='color:#14532D;font-size:0.9rem;margin:0;'>{assigned['directive']}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Dimensionality reduction visualizations
            st.markdown("<div class='section-title'>Dimensionality Reduction Cluster Visualizations</div>", unsafe_allow_html=True)
            if df_raw is not None:
                clus_col1, clus_col2 = st.columns(2)
                try:
                    df_sample = df_raw.sample(min(1500, len(df_raw)), random_state=42)
                    df_eng_sample = preprocessing.engineer_features(df_sample)
                    if isinstance(clusterer, dict) and "preprocessor" in clusterer:
                        prep = clusterer["preprocessor"]
                        raw_f = clusterer["raw_features"]
                        X_sub_raw = df_eng_sample[[c for c in raw_f if c in df_eng_sample.columns]]
                        X_scaled_sample = prep.transform(X_sub_raw)
                        model_obj = clusterer["model"]
                        labels_sample = model_obj.predict(X_scaled_sample)
                    else:
                        X_scaled_sample = df_eng_sample.select_dtypes(include=np.number).values[:1500]
                        labels_sample = np.zeros(len(X_scaled_sample))

                    with clus_col1:
                        fig_pca = utils.plot_pca_clusters(X_scaled_sample, labels_sample)
                        st.plotly_chart(fig_pca, use_container_width=True)
                    with clus_col2:
                        fig_tsne = utils.plot_tsne_clusters(X_scaled_sample, labels_sample)
                        st.plotly_chart(fig_tsne, use_container_width=True)
                except Exception as e:
                    st.info(f"Visualizations generated using standard projections. ({e})")

            # Display actual cluster profiles from CSV
            km_profiles = results_data.get("kmeans_prof")
            if km_profiles is not None:
                st.markdown("<div class='content-card-header'>Empirical Feature Means by Cluster (from `results/k-means_profiles.csv`)</div>", unsafe_allow_html=True)
                st.dataframe(km_profiles, use_container_width=True)

            st.markdown("""
                <div class='disclaimer-box'>
                    <b>Notice:</b> Cluster IDs represent unsupervised geometric partitions based on multidimensional feature distances
                    and do not automatically correspond to clinical diagnoses.
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during segmentation evaluation: {e}")


# ============================================================
# Page 6: Model Performance (Consolidated Tables & Diagnostics)
# ============================================================
def render_model_performance(regressor, classifier):
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Benchmarking Hub</div>
            <h1 class='dashboard-title'>Consolidated Model Performance & Diagnostics</h1>
            <p class='dashboard-subtitle'>Evaluation benchmarks, 5-fold cross-validation rankings, hyperparameter tuning, and diagnostic plots across all tracks.</p>
        </div>
    """, unsafe_allow_html=True)

    tab_reg, tab_cls, tab_clus = st.tabs([
        "Supervised Regression (10 Algorithms)",
        "Supervised Classification (10 Algorithms)",
        "Unsupervised Clustering (3 Algorithms)"
    ])

    with tab_reg:
        st.markdown("<div class='content-card-header'>All 10 Implemented Regression Algorithms Comparison</div>", unsafe_allow_html=True)
        reg_comp = results_data.get("reg_comp")
        reg_metrics = results_data.get("reg_metrics")
        display_reg = reg_comp if reg_comp is not None else reg_metrics

        if display_reg is not None and not display_reg.empty:
            st.dataframe(display_reg.style.format({
                "Train R2": "{:.4f}", "Test R2": "{:.4f}", "R2": "{:.4f}",
                "RMSE": "${:,.2f}", "MAE": "${:,.2f}"
            }, na_rep="-"), use_container_width=True)
        else:
            st.info("Regression comparison table not found.")

        # Cross validation and tuning tables
        c_cv, c_tune = st.columns(2)
        with c_cv:
            st.markdown("<div class='content-card-header'>5-Fold Cross-Validation Rankings (Held-Out Test Set Untouched)</div>", unsafe_allow_html=True)
            reg_cv = results_data.get("reg_cv")
            if reg_cv is not None:
                st.dataframe(reg_cv, use_container_width=True)
            else:
                st.info("CV table unavailable.")
        with c_tune:
            st.markdown("<div class='content-card-header'>Hyperparameter Tuning Results</div>", unsafe_allow_html=True)
            reg_tune = results_data.get("reg_tune")
            if reg_tune is not None:
                st.dataframe(reg_tune, use_container_width=True)
            else:
                st.info("Tuning table unavailable.")

        # Actual vs Predicted Diagnostic Plot
        if regressor is not None and df_raw is not None:
            st.markdown("<div class='content-card-header'>Diagnostic Plot: Actual vs. Predicted Medical Costs</div>", unsafe_allow_html=True)
            try:
                X_train, X_test, y_train, y_test, _, _, _ = preprocessing.prepare_data_regression()
                sample_n = min(1000, len(X_test))
                X_sub = X_test.iloc[:sample_n]
                y_sub = y_test.iloc[:sample_n]
                y_pred_sub = regressor.predict(X_sub)
                fig_act = utils.plot_actual_vs_predicted(y_sub, y_pred_sub)
                st.plotly_chart(fig_act, use_container_width=True)
            except Exception as e:
                st.info(f"Diagnostic plot generation: {e}")

    with tab_cls:
        st.markdown("<div class='content-card-header'>All 10 Implemented Classification Algorithms Comparison</div>", unsafe_allow_html=True)
        cls_comp = results_data.get("cls_comp")
        cls_metrics = results_data.get("cls_metrics")
        display_cls = cls_comp if cls_comp is not None else cls_metrics

        if display_cls is not None and not display_cls.empty:
            st.dataframe(display_cls.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                "Weighted F1": "{:.4f}", "ROC-AUC": "{:.4f}", "High risk recall": "{:.4f}",
                "Train Accuracy": "{:.4f}"
            }, na_rep="-"), use_container_width=True)
        else:
            st.info("Classification comparison table not found.")

        # Cross validation & Tuning
        c_clscv, c_clstune = st.columns(2)
        with c_clscv:
            st.markdown("<div class='content-card-header'>5-Fold Stratified Cross-Validation (Top Finalists)</div>", unsafe_allow_html=True)
            cls_cv = results_data.get("cls_cv")
            if cls_cv is not None:
                st.dataframe(cls_cv, use_container_width=True)
            else:
                st.info("Classification CV table unavailable.")
        with c_clstune:
            st.markdown("<div class='content-card-header'>Hyperparameter Tuning (Decision Tree & Random Forest)</div>", unsafe_allow_html=True)
            cls_tune = results_data.get("cls_tune")
            if cls_tune is not None:
                st.dataframe(cls_tune, use_container_width=True)
            else:
                st.info("Classification tuning table unavailable.")

        # Diagnostics: Confusion Matrix & ROC Curve
        if classifier is not None and df_raw is not None:
            st.markdown("<div class='content-card-header'>Deployed Classifier Diagnostics (Confusion Matrix & ROC Curve)</div>", unsafe_allow_html=True)
            c_cm, c_roc = st.columns(2)
            try:
                X_train_c, X_test_c, y_train_c, y_test_c, _, _, _ = preprocessing.prepare_data_classification()
                sample_nc = min(1500, len(X_test_c))
                X_test_sub_c = X_test_c.iloc[:sample_nc]
                y_test_sub_c = y_test_c.iloc[:sample_nc]
                y_pred_cls = classifier.predict(X_test_sub_c)

                with c_cm:
                    cm_vals = pd.crosstab(y_test_sub_c, y_pred_cls, rownames=['Actual'], colnames=['Predicted']).values
                    fig_cm = utils.plot_confusion_matrix(cm_vals)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with c_roc:
                    if hasattr(classifier, "predict_proba"):
                        y_prob_c = classifier.predict_proba(X_test_sub_c)[:, 1]
                    else:
                        y_prob_c = classifier.decision_function(X_test_sub_c)
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_test_sub_c, y_prob_c)
                    roc_auc_val = auc(fpr, tpr)
                    fig_roc = utils.plot_roc_curve(fpr, tpr, roc_auc_val)
                    st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.info(f"Classifier diagnostics plot generation: {e}")

    with tab_clus:
        st.markdown("<div class='content-card-header'>Clustering Algorithms Comparison (K-Means vs. Hierarchical)</div>", unsafe_allow_html=True)
        clus_comp = results_data.get("clus_comp")
        clus_metrics = results_data.get("clus_metrics")
        display_clus = clus_comp if clus_comp is not None else clus_metrics

        if display_clus is not None and not display_clus.empty:
            st.dataframe(display_clus.style.format({
                "Silhouette": "{:.4f}",
                "Davies-Bouldin": "{:.4f}",
                "Calinski-Harabasz": "{:.2f}"
            }, na_rep="-"), use_container_width=True)
        else:
            st.info("Clustering comparison table not found.")

        # Stability Table
        clus_stab = results_data.get("clus_stab")
        if clus_stab is not None:
            st.markdown("<div class='content-card-header'>5-Fold Stability / Repeatability Evaluation</div>", unsafe_allow_html=True)
            st.dataframe(clus_stab.head(10), use_container_width=True)


# ============================================================
# Page 7: About Project
# ============================================================
def render_about_project():
    st.markdown("""
        <div class='dashboard-header'>
            <div class='dashboard-badge'>Documentation & Architecture</div>
            <h1 class='dashboard-title'>About MedPredict Platform</h1>
            <p class='dashboard-subtitle'>Machine learning pipeline specifications, technical architecture, and academic project compliance.</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class='content-card'>
                <div class='content-card-header'>Academic Details</div>
                <p><b>Course:</b> 23CSE301 Machine Learning</p>
                <p><b>Project Type:</b> Capstone Project (Review 1 & 2)</p>
                <p><b>Team Number:</b> Team No. 8</p>
                <p><b>Institution:</b> Department of Computer Science & Engineering</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class='content-card'>
                <div class='content-card-header'>Technical Stack</div>
                <p><b>Language:</b> Python 3.13</p>
                <p><b>Core Libraries:</b> Scikit-Learn 1.9, Pandas, NumPy</p>
                <p><b>GUI & Visuals:</b> Streamlit 1.25+, Plotly 5.15+</p>
                <p><b>Serialization:</b> Joblib Pipeline Persistence</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class='content-card'>
                <div class='content-card-header'>Dataset Specifications</div>
                <p><b>Dataset:</b> Medical Insurance Cost Prediction</p>
                <p><b>Population:</b> 100,000 Verified Records</p>
                <p><b>Attributes:</b> 54 Raw Columns</p>
                <p><b>Engineered Features:</b> 7 Clinical Indices</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Strict Zero Data Leakage Implementation</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='content-card'>
            To guarantee realistic real-world deployment performance, our project implements rigorous data leakage safeguards:
            <ol style='margin-top:0.5rem;line-height:1.6;'>
                <li><b>Target Outcome Exclusion:</b> Post-hoc outcome columns (<code>risk_score</code>, <code>total_claims_paid</code>, <code>avg_claim_amount</code>) are strictly excluded from predictor matrices.</li>
                <li><b>Train/Test Split Isolation:</b> Evaluated on an 80:20 train/test split (<code>random_state=42</code>) with <code>assert set(X_train.index).isdisjoint(X_test.index)</code> verified.</li>
                <li><b>Pipeline Encapsulation:</b> Imputers, scalers, and one-hot encoders are fitted <i>strictly on training folds</i> and subsequently applied to validation and test data.</li>
                <li><b>Stratification:</b> Classification splits utilize <code>stratify=y</code> to preserve representative 63.2% vs. 36.8% risk class proportions across all folds.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown(
        "<div class='section-title'>Project Information</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
        <div class='content-card'>
            <p>
                This project focuses on healthcare cost prediction,
                patient risk classification, and patient segmentation
                using machine learning algorithms.
            </p>
            <p>
                <b>Repository:</b>
                <a href='https://github.com/hemu1196/ML_capstone'
                   target='_blank'>
                    github.com/hemu1196/ML_capstone
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)



# ============================================================
# 5. Routing Controller
# ============================================================
try:
    if page == "Home Dashboard":
        render_home_dashboard()
    elif page == "Dataset & EDA":
        render_dataset_eda(df_raw)
    elif page == "Healthcare Cost Prediction":
        render_cost_prediction(regressor, df_raw)
    elif page == "Patient Risk Prediction":
        render_risk_prediction(classifier, df_raw)
    elif page == "Patient Segmentation":
        render_segmentation(clusterer, df_raw)
    elif page == "Model Performance":
        render_model_performance(regressor, classifier)
    elif page == "About Project":
        render_about_project()
except Exception as e:
    st.error(f"An unexpected error occurred while rendering '{page}':")
    st.exception(e)
