import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Fix imports when running app directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.config as config
import app.preprocessing as preprocessing
import app.prediction as prediction
import app.utils as utils

# Page Configuration
st.set_page_config(
    page_title="MedIntel | Patient Risk & Cost Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS styling (Slate Slate Blue + Corporate Teal Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header & Titles */
    .dashboard-title {
        color: #0F172A;
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
    }
    
    .section-subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        border-color: #CBD5E1;
    }
    
    .metric-label {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    /* Badges for Classification results */
    .badge-high {
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #FEE2E2;
    }
    
    .badge-medium {
        background-color: #FFFBEB;
        color: #92400E;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #FEF3C7;
    }
    
    .badge-low {
        background-color: #F0FDF4;
        color: #166534;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #DCFCE7;
    }
    
    /* Button Customization */
    .stButton>button {
        background-color: #0F172A;
        color: #FFFFFF;
        border: 1px solid #0F172A;
        padding: 0.5rem 1.75rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.95rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1E293B;
        border-color: #1E293B;
        color: #FFFFFF;
    }
    
    /* Section Separation and Container styling */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .content-box-header {
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Caching Data & Models
@st.cache_resource
def load_models_cached():
    try:
        reg, cls = prediction.load_models()
        cluster_model_path = os.path.join(config.MODEL_DIR, "best_clustering.joblib")
        clus = joblib.load(cluster_model_path) if os.path.exists(cluster_model_path) else None
        return reg, cls, clus
    except Exception:
        return None, None, None

@st.cache_data
def load_data_cached():
    if os.path.exists(config.DATA_PATH):
        return pd.read_csv(config.DATA_PATH)
    return None

@st.cache_data
def load_metrics_cached():
    reg_metrics = pd.read_csv(config.REGRESSION_METRICS_PATH) if os.path.exists(config.REGRESSION_METRICS_PATH) else None
    cls_metrics = pd.read_csv(config.CLASSIFICATION_METRICS_PATH) if os.path.exists(config.CLASSIFICATION_METRICS_PATH) else None
    clus_metrics = pd.read_csv(os.path.join(config.RESULTS_DIR, "clustering_metrics.csv")) if os.path.exists(os.path.join(config.RESULTS_DIR, "clustering_metrics.csv")) else None
    return reg_metrics, cls_metrics, clus_metrics

df_raw = load_data_cached()
regressor, classifier, clusterer = load_models_cached()
reg_metrics, cls_metrics, clus_metrics = load_metrics_cached()

# Sidebar Navigation Configuration
st.sidebar.markdown("<h2 style='color:#0F172A;font-weight:700;font-size:1.4rem;margin-bottom:1.5rem;'>MedIntel Insights</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    ["Home", "Dashboard", "Cost Prediction", "Risk Prediction", "Patient Segmentation", "Model Performance", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:0.75rem;color:#94A3B8;'>Authorized Access Only.<br>HIPAA Compliance Enforced.</p>", unsafe_allow_html=True)

# Routing
if page == "Home":
    st.markdown("<div class='dashboard-title'>Clinical Diagnostics & Financial Risk Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>AI-Driven Medical Claims Forecasting and Patient Risk Intelligence Dashboard</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Population</div><div class='metric-value'>100,000</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Features</div><div class='metric-value'>54</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-label'>Regression Algorithms</div><div class='metric-value'>10</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-label'>Classifiers Deployed</div><div class='metric-value'>5</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown("<div class='metric-card'><div class='metric-label'>Cluster Cohorts</div><div class='metric-value'>4</div></div>", unsafe_allow_html=True)

    st.markdown("### Executive Summary & Scope")
    st.markdown("""
    This analytics platform supports clinical decision-making, actuarial risk assessment, and financial modeling for healthcare populations. 
    It is divided into three key quantitative tracks:
    * **Track 1: Cost Regression** - Forecasts individual patient annual medical costs based on demographic, lifestyle, and clinical parameters.
    * **Track 2: Risk Classification** - Classifies patients into high-risk categories to identify candidates for early preventative health intervention.
    * **Track 3: Patient Segmentation** - Discovers latent patient cohorts using clustering algorithms based on healthcare resource utilization and chronic disease count.
    """)
    
    st.markdown("### Process Pipeline and Workflow")
    st.markdown("""
    ```mermaid
    graph TD
        A[Kaggle Dataset: 100k Rows] --> B[Data Preprocessing & Auditing]
        B --> C[Clinical Feature Engineering]
        C --> D[Track 1: Cost Regression - 10 Models]
        C --> E[Track 2: Risk Classification Part A - 5 Models]
        C --> F[Track 3: Patient Segmentation - KMeans]
        D --> G[Best Regressor Saved: Tuned Random Forest]
        E --> H[Best Classifier Saved: Tuned Decision Tree]
        F --> I[Cohort Definitions Saved]
        G --> J[MedIntel Interactive Dashboard Web App]
        H --> J
        I --> J
    ```
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    st.markdown("<div class='dashboard-title'>Clinical Population Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Statistical Summaries and Distributions of the Covered Population</div>", unsafe_allow_html=True)
    
    if df_raw is not None:
        avg_cost = df_raw["annual_medical_cost"].mean()
        avg_bmi = df_raw["bmi"].mean()
        avg_income = df_raw["income"].mean()
        high_risk_pct = df_raw["is_high_risk"].mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Average Medical Cost</div><div class='metric-value'>${avg_cost:,.2f}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Average BMI</div><div class='metric-value'>{avg_bmi:.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Average Income</div><div class='metric-value'>${avg_income:,.2f}</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>High Risk Patients</div><div class='metric-value'>{high_risk_pct:.2f}%</div></div>", unsafe_allow_html=True)
                
        st.markdown("### Financial & Risk Distribution Analysis")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            fig_cost = utils.plot_cost_distribution(df_raw.sample(5000))
            st.plotly_chart(fig_cost, use_container_width=True)
        with row1_col2:
            fig_risk = utils.plot_risk_distribution(df_raw)
            st.plotly_chart(fig_risk, use_container_width=True)
            
        st.markdown("### Feature Interactions and Importances")
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            num_cols_corr = ["age", "bmi", "income", "visits_last_year", "medication_count", "annual_medical_cost"]
            fig_corr = utils.plot_correlation_heatmap(df_raw, num_cols_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
        with row2_col2:
            if regressor is not None:
                fig_imp = utils.plot_feature_importance(regressor, top_n=10)
                if fig_imp is not None:
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Feature importance plotting not supported by selected estimator.")
            else:
                st.warning("Model must be trained to extract feature importances.")
    else:
        st.error("Kaggle dataset not found. Verify file path configuration.")

else:
    if page in ["Cost Prediction", "Risk Prediction"]:
        st.markdown(f"<div class='dashboard-title'>{page}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Enter Patient Profile Details below for Model Scoring</div>", unsafe_allow_html=True)
        
        # Grid input style
        st.markdown("<div class='content-box-header'>Demographics and Social Determinants of Health</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=45)
            sex = st.selectbox("Gender / Sex", ["Female", "Male", "Other"])
            bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=28.4, step=0.1)
        with col2:
            income = st.number_input("Annual Income ($)", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
            smoker = st.selectbox("Tobacco / Smoking History", ["Never", "Current", "Former"])
            alcohol = st.selectbox("Alcohol Consumption Frequency", ["None", "Occasional", "Weekly", "Daily"])
        with col3:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
            region = st.selectbox("Residential Region", ["North", "Central", "West", "South", "East"])
            urban_rural = st.selectbox("Residential Classification", ["Suburban", "Urban", "Rural"])
        with col4:
            education = st.selectbox("Highest Education Level", ["HS", "No HS", "Some College", "Bachelors", "Masters", "Doctorate"])
            employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Retired", "Unemployed"])
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])

        st.markdown("<div class='content-box-header'>Clinical Vitals and Chronic Diagnoses</div>", unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            diabetes = st.selectbox("Diabetes Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
            hypertension = st.selectbox("Hypertension Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
            heart_disease = st.selectbox("Cardiovascular Disease Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
        with col6:
            asthma = st.selectbox("Asthma Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
            copd = st.selectbox("COPD Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
            kidney = st.selectbox("Chronic Kidney Disease Status", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
        with col7:
            visits = st.number_input("Outpatient Office Visits (Past Year)", min_value=0, max_value=50, value=1)
            systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=130)
            diastolic = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=85)
        with col8:
            plan = st.selectbox("Insurance Plan Structure", ["HMO", "PPO", "EPO", "POS"])
            hba1c = st.number_input("Fasting HbA1c (%)", min_value=3.5, max_value=15.0, value=5.6, step=0.1)
            had_procedure = st.selectbox("Major Surgery History", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

        # Map to Patient Dict structure (inferred features)
        hosp_3yrs = 1 if visits > 2 else 0
        days_hosp = hosp_3yrs * 3
        med_count = 2 if (diabetes + hypertension) > 0 else 0
        ldl = 110.0
        network_tier = "Silver"
        ded = 1000.0
        copay_val = 20.0
        policy_term = 1.0
        policy_changes = 0.0
        provider_quality = 3.6
        annual_premium = 1500.0
        monthly_premium = 125.0
        claims_count = visits
        avg_claim = 400.0
        total_claims = claims_count * avg_claim
        
        patient_dict = {
            "age": age, "sex": sex, "region": region, "urban_rural": urban_rural, "income": income,
            "education": education, "marital_status": marital, "employment_status": employment,
            "household_size": dependents + 2, "dependents": dependents, "bmi": bmi, "smoker": smoker,
            "alcohol_freq": alcohol if alcohol != "None" else np.nan, "visits_last_year": visits,
            "hospitalizations_last_3yrs": hosp_3yrs, "days_hospitalized_last_3yrs": days_hosp,
            "medication_count": med_count, "systolic_bp": systolic, "diastolic_bp": diastolic,
            "ldl": ldl, "hba1c": hba1c, "plan_type": plan, "network_tier": network_tier,
            "deductible": ded, "copay": copay_val, "policy_term_years": policy_term,
            "policy_changes_last_2yrs": policy_changes, "provider_quality": provider_quality,
            "annual_premium": annual_premium, "monthly_premium": monthly_premium, "claims_count": claims_count,
            "avg_claim_amount": avg_claim, "total_claims_paid": total_claims, "chronic_count": diabetes + hypertension + heart_disease,
            "hypertension": hypertension, "diabetes": diabetes, "asthma": asthma, "copd": copd,
            "cardiovascular_disease": heart_disease, "cancer_history": 0, "kidney_disease": kidney,
            "liver_disease": 0, "arthritis": 0, "mental_health": 0, "proc_imaging_count": 0,
            "proc_surgery_count": 0, "proc_physio_count": 0, "proc_consult_count": 0, "proc_lab_count": 0,
            "had_major_procedure": had_procedure
        }

        if page == "Cost Prediction":
            if st.button("Generate Cost Forecast"):
                if regressor is not None:
                    with st.spinner("Executing regression pipelines..."):
                        predicted_cost = prediction.predict_cost(patient_dict)
                    
                    st.markdown("---")
                    st.markdown("### Cost Projections & Diagnostics")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown(f"""
                            <div style='background-color:#F8FAFC; border: 1px solid #E2E8F0; border-left: 6px solid #0F172A; border-radius: 8px; padding: 2rem;'>
                                <h4 style='color:#64748B; margin:0; font-size: 0.95rem; font-weight:600; text-transform:uppercase;'>Predicted Annual Medical Liability</h4>
                                <h1 style='color:#0F172A; font-size:3.2rem; margin-top:0.5rem;'>${predicted_cost:,.2f}</h1>
                                <p style='color:#94A3B8; font-size:0.85rem; margin-top:0.5rem; margin-bottom:0;'>Estimates assume regular outpatient care and active management.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        fig_gauge = utils.plot_cost_gauge(predicted_cost)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    with res_col2:
                        fig_contrib = utils.plot_waterfall_contribution(patient_dict)
                        st.plotly_chart(fig_contrib, use_container_width=True)
                else:
                    st.error("Saved Regressor Model not found. Run model training notebook first.")

        elif page == "Risk Prediction":
            if st.button("Evaluate Risk Cohort"):
                if classifier is not None:
                    with st.spinner("Executing classifier pipelines..."):
                        risk_results = prediction.predict_risk(patient_dict)
                    
                    st.markdown("---")
                    st.markdown("### Risk Cohort Classification & Recommendations")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        bg_color = "#FEF2F2" if risk_results["risk_level"] == "High" else ("#FFFBEB" if risk_results["risk_level"] == "Medium" else "#F0FDF4")
                        border_color = "#FEE2E2" if risk_results["risk_level"] == "High" else ("#FEF3C7" if risk_results["risk_level"] == "Medium" else "#DCFCE7")
                        text_color = "#991B1B" if risk_results["risk_level"] == "High" else ("#92400E" if risk_results["risk_level"] == "Medium" else "#166534")
                        
                        st.markdown(f"""
                            <div style='background-color:{bg_color}; border: 1px solid {border_color}; border-left: 6px solid {text_color}; border-radius: 8px; padding: 2rem;'>
                                <h4 style='color:#64748B; margin:0; font-size: 0.95rem; font-weight:600; text-transform:uppercase;'>Assigned Risk Cohort</h4>
                                <h1 style='color:{text_color}; font-size:2.8rem; margin-top:0.5rem;'>{risk_results["risk_level"]} Risk Status</h1>
                                <p style='color:{text_color}; font-size:1rem; margin-top:0.5rem; margin-bottom:0;'>Cohort Probability Score: <b>{risk_results["probability"]*100:.1f}%</b></p>
                            </div>
                        """, unsafe_allow_html=True)
                        fig_meter = utils.plot_risk_meter(risk_results["probability"])
                        st.plotly_chart(fig_meter, use_container_width=True)
                    with res_col2:
                        st.markdown("#### Clinical Directives & Preventive Care Plan")
                        for idx, rec in enumerate(risk_results["recommendations"]):
                            st.info(f"Directive #{idx+1}: {rec}")
                else:
                    st.error("Saved Classifier Model not found. Run model training notebook first.")

    elif page == "Patient Segmentation":
        st.markdown("<div class='dashboard-title'>Patient Cohort Segmentation</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Assign Patient Profiles to Lifestyle and Cost-Risk Cohorts (K-Means Clustering)</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='content-box-header'>Profile Metrics for Cluster Assessment</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=58)
            bmi = st.number_input("Patient BMI", min_value=15.0, max_value=50.0, value=34.2)
        with col2:
            income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=300000.0, value=65000.0)
            visits = st.number_input("Outpatient Hospital Visits (Past Year)", min_value=0, max_value=20, value=3)
        with col3:
            diabetes = st.checkbox("Diabetes Diagnosis", value=True)
            hypertension = st.checkbox("Hypertension Diagnosis", value=True)
            heart_disease = st.checkbox("Cardiovascular Disease Diagnosis", value=False)
            smoker = st.selectbox("Smoking Status Profile", ["Current", "Never", "Former"])
            
        if st.button("Evaluate Patient Segmentation"):
            if clusterer is not None:
                total_chronic = sum([int(diabetes), int(hypertension), int(heart_disease)])
                util_score = visits * 2.0
                smoker_weight = 3.0 if smoker == "Current" else (1.0 if smoker == "Former" else 0.0)
                lifestyle_score = smoker_weight + (total_chronic * 0.5)
                
                feat_df = pd.DataFrame([{
                    "Age": age,
                    "BMI": bmi,
                    "Income": income,
                    "Hospital_Utilization_Score": util_score,
                    "Total_Chronic_Diseases": total_chronic,
                    "Lifestyle_Risk_Score": lifestyle_score
                }])
                
                scaler = clusterer["scaler"]
                model = clusterer["model"]
                
                feat_scaled = scaler.transform(feat_df)
                cluster_id = model.predict(feat_scaled)[0]
                
                cohorts = {
                    0: {"name": "High Lifestyle Risk Cohort", "lifestyle": "Elevated. Heavy tobacco usage combined with high BMI.", "disease": "Borderline blood pressure and glucose markers.", "cost": "Moderate-High yearly claims.", "insight": "Prioritize smoking cessation and dietary coordination.", "color": "#D97706"},
                    1: {"name": "Low-Risk Well Cohort", "lifestyle": "Excellent. Highly active, non-smokers, normal weight.", "disease": "No active diagnoses.", "cost": "Low claims profile.", "insight": "Support ongoing health maintenance and annual wellness checks.", "color": "#059669"},
                    2: {"name": "Geriatric Chronic Care Segment", "lifestyle": "Sedentary. Restricted activity patterns.", "disease": "Multiple complex chronic comorbidities.", "cost": "Very High claims profile.", "insight": "Require direct care management and outreach protocols.", "color": "#DC2626"},
                    3: {"name": "Moderate-Risk Managed Cohort", "lifestyle": "Average. Moderate exercise activity.", "disease": "Single chronic condition managed via maintenance drugs.", "cost": "Predictable moderate claims.", "insight": "Implement medication compliance checks.", "color": "#2563EB"}
                }
                
                selected_cohort = cohorts[cluster_id]
                
                st.markdown("---")
                st.markdown(f"### Assigned Cohort Segment: <span style='color:{selected_cohort['color']};'>{selected_cohort['name']}</span>", unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                        <div style='background-color:#F8FAFC; border-radius: 8px; padding: 1.5rem; border: 1px solid #E2E8F0; border-top: 5px solid {selected_cohort['color']}'>
                            <p><b>Lifestyle Profile:</b> {selected_cohort['lifestyle']}</p>
                            <p><b>Disease Profile:</b> {selected_cohort['disease']}</p>
                            <p><b>Medical Cost Category:</b> {selected_cohort['cost']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                        <div style='background-color:#F0FDF4; border-radius: 8px; padding: 1.5rem; border: 1px solid #DCFCE7; border-top: 5px solid #166534'>
                            <h4 style='margin:0;color:#166534;font-size:1.05rem;font-weight:600;'>Care Directives</h4>
                            <p style='margin-top:0.5rem;color:#14532D;font-size:0.95rem;'>{selected_cohort['insight']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Cluster model not found. Run training first.")

    elif page == "Model Performance":
        st.markdown("<div class='dashboard-title'>Model Performance Analytics</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Comparative Diagnostics of Deployed Estimators on Kaggle Test Splits</div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Cost Regression", "Patient Risk Classification", "Patient Clustering"])
        
        with tab1:
            st.markdown("#### Regression Algorithm Comparison Table")
            if reg_metrics is not None:
                st.dataframe(reg_metrics.style.highlight_max(subset=["R2"], color="#DCFCE7"), use_container_width=True)
                fig_reg_comp = px.bar(
                    reg_metrics,
                    x="R2",
                    y="Model",
                    orientation="h",
                    color="R2",
                    color_continuous_scale="Blues",
                    title="Model R² Score Comparison (Test Set)",
                    template="plotly_white"
                )
                fig_reg_comp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_reg_comp, use_container_width=True)
            else:
                st.warning("Metrics database not found.")
                
        with tab2:
            st.markdown("#### Classification Algorithm Comparison Table")
            if cls_metrics is not None:
                st.dataframe(cls_metrics.style.highlight_max(subset=["F1-Score"], color="#DCFCE7"), use_container_width=True)
                fig_cls_comp = px.bar(
                    cls_metrics,
                    x="F1-Score",
                    y="Model",
                    orientation="h",
                    color="F1-Score",
                    color_continuous_scale="Purples",
                    title="Model F1-Score Comparison (Test Set)",
                    template="plotly_white"
                )
                fig_cls_comp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_cls_comp, use_container_width=True)
            else:
                st.warning("Metrics database not found.")
                
        with tab3:
            st.markdown("#### Clustering Algorithm Metrics")
            if clus_metrics is not None:
                st.table(clus_metrics)

    elif page == "About Project":
        st.markdown("<div class='dashboard-title'>System Architecture & Developer Details</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Technical Pipeline and Framework Summary</div>", unsafe_allow_html=True)
        
        st.markdown("### Technical Framework Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Data Processing & Preprocessing**
            - Python 3.13 Development
            - Pandas & NumPy for data cleaning
            - Scikit-Learn pipelines
            """)
        with col2:
            st.markdown("""
            **GUI Framework & Visualization**
            - Streamlit for dashboard rendering
            - Plotly for interactive distributions
            - Joblib for model serialization
            """)
        with col3:
            st.markdown("""
            **Scoring Estimators**
            - Track 1: 10 Regression Models
            - Track 2: 5 Classifiers
            - Track 3: K-Means Clustering
            """)
            
        st.markdown("### Developer Profiles")
        st.markdown("""
        Project completed under Course: 23CSE301 Machine Learning - Capstone Project.
        
        * **GitHub Repository**: [github.com/hemu1196/ML_capstone](https://github.com/hemu1196/ML_capstone)
        * **LinkedIn Profile**: [linkedin.com/in/hemachandra](https://linkedin.com)
        """)
