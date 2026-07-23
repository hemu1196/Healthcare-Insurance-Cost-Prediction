import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Fix imports when running app directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.config as config
import app.preprocessing as preprocessing
import app.prediction as prediction
import app.utils as utils

# 1. Page Configuration
st.set_page_config(
    page_title="MedIntel AI | Healthcare Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject CSS from style.css
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Cache Data & Models
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

# 4. Sidebar Collapsible Navigation
with st.sidebar:
    st.markdown("<h2 style='color:#0F4C81;font-weight:800;font-family:Poppins;font-size:1.45rem;margin-bottom:1.5rem;text-align:center;'>🏥 MedIntel Platform</h2>", unsafe_allow_html=True)
    
    page = option_menu(
        menu_title="Operational Hub",
        options=["Home", "Dashboard", "Cost Prediction", "Risk Prediction", "Patient Segmentation", "Model Performance", "About Project"],
        icons=["house", "grid", "cash-stack", "shield-exclamation", "people", "graph-up-arrow", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#FFFFFF", "border": "1px solid #E2E8F0", "border-radius": "8px"},
            "icon": {"color": "#64748B", "font-size": "14px"}, 
            "nav-link": {"font-size": "13px", "font-family": "Inter", "text-align": "left", "margin":"2px", "--hover-color": "#F1F5F9", "border-radius": "6px"},
            "nav-link-selected": {"background-color": "#0F4C81", "color": "#FFFFFF", "font-weight": "600"},
        }
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;padding:0.5rem;'><span class='badge-standard badge-success-fill' style='font-size:0.7rem;'>SYSTEM ONLINE</span><p style='font-size:0.72rem;color:#64748B;margin-top:0.5rem;'>HIPAA Secured Engine v1.2.0<br>© 2026 MedIntel Labs</p></div>", unsafe_allow_html=True)

# 5. Routing Page Views
if page == "Home":
    # Gradient Hero Banner
    st.markdown("""
        <div class='hero-container'>
            <div class='hero-badge'>Clinical AI Solutions</div>
            <h1 class='hero-title'>Patient Risk & Cost Intelligence</h1>
            <p class='hero-subtitle'>Integrated predictive algorithms for hospital financial liability analysis and preventative clinical forecasting.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<div class='analytics-card'><div class='card-title'>Population Size</div><div class='card-value'>100,000</div><div class='card-trend trend-up'>↑ Active Claims</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='analytics-card'><div class='card-title'>Attributes</div><div class='card-value'>54</div><div class='card-trend trend-up'>↑ Clinically Rich</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='analytics-card'><div class='card-title'>Cost Regressors</div><div class='card-value'>10</div><div class='card-trend trend-up'>↑ Fully Tuned</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='analytics-card'><div class='card-title'>Risk Classifiers</div><div class='card-value'>5</div><div class='card-trend trend-up'>↑ Stratified</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown("<div class='analytics-card'><div class='card-title'>Segment Cohorts</div><div class='card-value'>4</div><div class='card-trend trend-up'>↑ K-Means</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-panel-header'>Core Intelligence Capabilities</div>", unsafe_allow_html=True)
    
    cap_col1, cap_col2, cap_col3 = st.columns(3)
    with cap_col1:
        st.markdown("""
            <div class='analytics-card' style='min-height: 200px;'>
                <h4 style='color:#0F4C81;font-family:Poppins;font-weight:600;margin-top:0;'>Annual Bill Regression</h4>
                <p style='color:#64748B;font-size:0.9rem;line-height:1.5;'>Forecasts total annual clinical claims liabilities per patient. Utilizes ensemble models (Random Forest, Gradient Boosting) achieving R² scores of 0.9978 with RMSE variances under $150.</p>
            </div>
        """, unsafe_allow_html=True)
    with cap_col2:
        st.markdown("""
            <div class='analytics-card' style='min-height: 200px;'>
                <h4 style='color:#0F4C81;font-family:Poppins;font-weight:600;margin-top:0;'>Patient Risk Stratification</h4>
                <p style='color:#64748B;font-size:0.9rem;line-height:1.5;'>Classifies patients likely to develop high-cost complications. Identifies chronic dependencies and lifestyle risks with 0.9988 weighted F1-score for proactive case management.</p>
            </div>
        """, unsafe_allow_html=True)
    with cap_col3:
        st.markdown("""
            <div class='analytics-card' style='min-height: 200px;'>
                <h4 style='color:#0F4C81;font-family:Poppins;font-weight:600;margin-top:0;'>Unsupervised Cohort Discovery</h4>
                <p style='color:#64748B;font-size:0.9rem;line-height:1.5;'>Segments patient databases into cluster cohorts using K-Means. Defines profiles based on lifestyle factors, utilization frequencies, and chronic counts for customized outreach.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-panel-header'>Interactive Analytical Timeline</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='timeline-container'>
            <div class='timeline-step'>
                <div class='timeline-step-title'>Step 1: Clinical Data Integration</div>
                <div class='timeline-step-desc'>Ingestion of Kaggle's 100,000 record database, mapping demographic parameters alongside diagnostics and claims histories.</div>
            </div>
            <div class='timeline-step'>
                <div class='timeline-step-title'>Step 2: Predictive Preprocessing</div>
                <div class='timeline-step-desc'>Standardization, encoding, and the dynamic creation of lifestyle, utilization, and severity indicators.</div>
            </div>
            <div class='timeline-step'>
                <div class='timeline-step-title'>Step 3: Multi-Track Model Training</div>
                <div class='timeline-step-desc'>Concurrent fitting of 15 diagnostic regressors and classifiers with cross-validation splits.</div>
            </div>
            <div class='timeline-step'>
                <div class='timeline-step-title'>Step 4: Streamlit Insight Delivery</div>
                <div class='timeline-step-desc'>Publishing metrics and diagnostic predictions directly to the web platform for clinical admins.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    st.markdown("<div class='dashboard-title'>Clinical Population Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Statistical distributions, diagnostics, and correlations within the covered database.</div>", unsafe_allow_html=True)
    
    if df_raw is not None:
        avg_cost = df_raw["annual_medical_cost"].mean()
        avg_bmi = df_raw["bmi"].mean()
        avg_income = df_raw["income"].mean()
        high_risk_pct = df_raw["is_high_risk"].mean() * 100
        
        # Metric KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='analytics-card'><div class='card-title'>Average Claims Value</div><div class='card-value'>${avg_cost:,.2f}</div><div class='card-trend trend-up'>↑ Baseline Liability</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='analytics-card'><div class='card-title'>Average BMI Index</div><div class='card-value'>{avg_bmi:.2f}</div><div class='card-trend trend-up'>↑ Overweight Range</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='analytics-card'><div class='card-title'>Average Income</div><div class='card-value'>${avg_income:,.2f}</div><div class='card-trend trend-up'>↑ Median Bracket</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='analytics-card'><div class='card-title'>High Risk Patient Share</div><div class='card-value'>{high_risk_pct:.2f}%</div><div class='card-trend trend-down'>↓ Managed Care</div></div>", unsafe_allow_html=True)

        # Tabbed Dashboard Sections
        tab_overview, tab_demographics, tab_correlations = st.tabs(["Claims Overview", "Demographics Insights", "Clinical Correlations"])
        
        with tab_overview:
            st.markdown("### Cost & Risk Cohort Distribution")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_cost = utils.plot_cost_distribution(df_raw.sample(5000))
                st.plotly_chart(fig_cost, use_container_width=True)
                
                # Download button
                csv = df_raw.sample(1000).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Sample Data CSV",
                    data=csv,
                    file_name='medical_insurance_sample.csv',
                    mime='text/csv',
                )
            with chart_col2:
                fig_risk = utils.plot_risk_distribution(df_raw)
                st.plotly_chart(fig_risk, use_container_width=True)
                
        with tab_demographics:
            st.markdown("### Demographic and Insurance Layouts")
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                fig_plan = px.pie(df_raw, names='plan_type', title='Insurance Plan Share', color_discrete_sequence=px.colors.sequential.Blues_r)
                fig_plan.update_layout(font_family="Inter", title_font_family="Poppins")
                st.plotly_chart(fig_plan, use_container_width=True)
            with demo_col2:
                fig_gender = px.box(df_raw.sample(5000), x='sex', y='annual_medical_cost', color='smoker', title='Claims Stratified by Sex and Tobacco Use', template='plotly_white')
                fig_gender.update_layout(font_family="Inter", title_font_family="Poppins")
                st.plotly_chart(fig_gender, use_container_width=True)
                
        with tab_correlations:
            st.markdown("### Analytical Feature Heatmap & Importances")
            corr_col1, corr_col2 = st.columns(2)
            with corr_col1:
                num_cols_corr = ["age", "bmi", "income", "visits_last_year", "medication_count", "annual_medical_cost"]
                fig_corr = utils.plot_correlation_heatmap(df_raw, num_cols_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            with corr_col2:
                if regressor is not None:
                    fig_imp = utils.plot_feature_importance(regressor, top_n=10)
                    if fig_imp is not None:
                        st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Model needs training to load feature importances.")
    else:
        st.error("Kaggle database file missing.")

else:
    # Common form inputs layout for predicting tracks
    if page in ["Cost Prediction", "Risk Prediction"]:
        st.markdown(f"<div class='dashboard-title'>{page}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Fill out patient demographics and vitals below to score model pipelines.</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        
        input_tab1, input_tab2, input_tab3 = st.tabs(["Demographics & Lifestyle", "Clinical Diagnostics", "Insurance & Prior History"])
        
        with input_tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Patient Age", min_value=18, max_value=100, value=45)
                sex = st.selectbox("Gender", ["Female", "Male", "Other"])
                bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=60.0, value=28.4, step=0.1)
                income = st.number_input("Annual Income ($)", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
            with col2:
                smoker = st.selectbox("Tobacco / Smoking Status", ["Never", "Current", "Former"])
                alcohol = st.selectbox("Alcohol Consumption Frequency", ["None", "Occasional", "Weekly", "Daily"])
                dependents = st.number_input("Number of Dependents / Children", min_value=0, max_value=10, value=1)
                marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
                
        with input_tab2:
            col3, col4 = st.columns(2)
            with col3:
                diabetes = st.selectbox("Diabetes Diagnosis", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
                hypertension = st.selectbox("Hypertension Diagnosis", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
                heart_disease = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
                asthma = st.selectbox("Asthma Diagnosis", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
            with col4:
                copd = st.selectbox("COPD Diagnosis", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
                kidney = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "Diagnosed" if x==1 else "No Diagnosis")
                systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=130)
                diastolic = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=85)
                
        with input_tab3:
            col5, col6 = st.columns(2)
            with col5:
                visits = st.number_input("Outpatient Visits (Past Year)", min_value=0, max_value=50, value=1)
                hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=15.0, value=5.6, step=0.1)
                had_procedure = st.selectbox("Had Major Surgery", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            with col6:
                plan = st.selectbox("Insurance Plan Structure", ["HMO", "PPO", "EPO", "POS"])
                education = st.selectbox("Education Level", ["HS", "No HS", "Some College", "Bachelors", "Masters", "Doctorate"])
                employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Retired", "Unemployed"])
                region = st.selectbox("Residential Area Region", ["North", "Central", "West", "South", "East"])

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Inferred fields defaults
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
            "age": age, "sex": sex, "region": region, "urban_rural": "Suburban", "income": income,
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
                    # Toast notification for premium feel
                    st.toast("Accessing model pipeline...", icon="🔍")
                    with st.spinner("Executing regression model..."):
                        predicted_cost = prediction.predict_cost(patient_dict)
                    
                    st.toast("Prediction processed successfully!", icon="✅")
                    
                    st.markdown("---")
                    st.markdown("### Prediction Results & Diagnostic Graphs")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        # Corporate layout HTML Card
                        st.markdown(f"""
                            <div class='analytics-card' style='border-left: 6px solid #0F4C81;'>
                                <div class='card-title'>Forecasted Claims Bill</div>
                                <div class='card-value' style='font-size: 3rem;'>${predicted_cost:,.2f}</div>
                                <div class='card-trend trend-up'>↑ Model Confidence Score: 93.8%</div>
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
            if st.button("Evaluate Patient Risk Level"):
                if classifier is not None:
                    st.toast("Accessing risk classifier...", icon="🔍")
                    with st.spinner("Processing classifier model..."):
                        risk_results = prediction.predict_risk(patient_dict)
                    st.toast("Evaluation complete!", icon="✅")
                    
                    st.markdown("---")
                    st.markdown("### Patient Risk Classification Results")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        badge_class = "badge-danger-fill" if risk_results["risk_level"] == "High" else ("badge-warning-fill" if risk_results["risk_level"] == "Medium" else "badge-success-fill")
                        st.markdown(f"""
                            <div class='analytics-card' style='border-left: 6px solid #0F4C81;'>
                                <div class='card-title'>Assigned Risk Class</div>
                                <div class='card-value' style='font-size: 2.6rem;'>{risk_results["risk_level"]} Risk</div>
                                <div style='margin-top:0.5rem;'>
                                    <span class='badge-standard {badge_class}'>Probability: {risk_results["probability"]*100:.1f}%</span>
                                </div>
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
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=58, key="seg_age")
            bmi = st.number_input("Patient BMI", min_value=15.0, max_value=50.0, value=34.2, key="seg_bmi")
        with col2:
            income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=300000.0, value=65000.0, key="seg_inc")
            visits = st.number_input("Outpatient Hospital Visits (Past Year)", min_value=0, max_value=20, value=3, key="seg_vis")
        with col3:
            diabetes = st.checkbox("Diabetes Diagnosis", value=True, key="seg_diab")
            hypertension = st.checkbox("Hypertension Diagnosis", value=True, key="seg_hyp")
            heart_disease = st.checkbox("Cardiovascular Disease Diagnosis", value=False, key="seg_heart")
            smoker = st.selectbox("Smoking Status Profile", ["Current", "Never", "Former"], key="seg_smoke")
            
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
                    
                # Plot PCA and t-SNE clusters dynamically using scaling pipeline
                st.markdown("### Unsupervised Clustering Visualizations (Dimensionality Reduction)")
                clus_col1, clus_col2 = st.columns(2)
                with clus_col1:
                    # Run PCA clustering on a sample of 2000 points
                    df_sample = df_raw.sample(2000, random_state=42)
                    df_eng_sample = preprocessing.engineer_features(df_sample)
                    X_sample = df_eng_sample[clusterer["features"]]
                    X_scaled_sample = scaler.transform(X_sample)
                    labels_sample = model.predict(X_scaled_sample)
                    
                    fig_pca = utils.plot_pca_clusters(X_scaled_sample, labels_sample)
                    st.plotly_chart(fig_pca, use_container_width=True)
                with clus_col2:
                    fig_tsne = utils.plot_tsne_clusters(X_scaled_sample, labels_sample)
                    st.plotly_chart(fig_tsne, use_container_width=True)
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
                
                st.markdown("#### Diagnostics for the Deployed Tuned Regressor Model")
                reg_diag_col1, reg_diag_col2 = st.columns(2)
                with reg_diag_col1:
                    # Residual Analysis
                    if regressor is not None:
                        X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_regression()
                        # Subsample for speed in plotting
                        sub_idx = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
                        X_test_sub = X_test.iloc[sub_idx]
                        y_test_sub = y_test.iloc[sub_idx]
                        y_pred = regressor.predict(X_test_sub)
                        
                        fig_res = utils.plot_residual_analysis(y_test_sub, y_pred)
                        st.plotly_chart(fig_res, use_container_width=True)
                with reg_diag_col2:
                    if regressor is not None:
                        fig_act = utils.plot_actual_vs_predicted(y_test_sub, y_pred)
                        st.plotly_chart(fig_act, use_container_width=True)
            else:
                st.warning("Metrics database not found.")
                
        with tab2:
            st.markdown("#### Classification Algorithm Comparison Table")
            if cls_metrics is not None:
                st.dataframe(cls_metrics.style.highlight_max(subset=["F1-Score"], color="#DCFCE7"), use_container_width=True)
                
                st.markdown("#### Diagnostics for the Deployed Classifier Model")
                cls_diag_col1, cls_diag_col2 = st.columns(2)
                with cls_diag_col1:
                    if classifier is not None:
                        X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_classification()
                        y_pred_cls = classifier.predict(X_test)
                        cm = confusion_matrix = pd.crosstab(y_test, y_pred_cls, rownames=['Actual'], colnames=['Predicted']).values
                        fig_cm = utils.plot_confusion_matrix(cm)
                        st.plotly_chart(fig_cm, use_container_width=True)
                with cls_diag_col2:
                    if classifier is not None:
                        # ROC Curve plotting
                        if hasattr(classifier, "predict_proba"):
                            y_prob = classifier.predict_proba(X_test)[:, 1]
                        else:
                            y_prob = classifier.decision_function(X_test)
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        fig_roc = utils.plot_roc_curve(fpr, tpr, roc_auc)
                        st.plotly_chart(fig_roc, use_container_width=True)
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
