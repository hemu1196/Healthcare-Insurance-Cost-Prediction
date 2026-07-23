import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_cost_distribution(df):
    """
    Generate a Plotly histogram/KDE for the distribution of medical costs.
    """
    fig = px.histogram(
        df, 
        x="annual_medical_cost", 
        color_discrete_sequence=["#1E88E5"],
        marginal="box",
        title="Distribution of Annual Medical Costs",
        labels={"annual_medical_cost": "Annual Medical Cost ($)"},
        template="plotly_white"
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Annual Medical Cost ($)",
        yaxis_title="Count",
        bargap=0.05
    )
    return fig

def plot_risk_distribution(df):
    """
    Generate a Plotly bar chart for patient risk levels.
    """
    counts = df["is_high_risk"].value_counts().reset_index()
    counts["Risk Label"] = counts["is_high_risk"].map({0: "Low/Medium Risk", 1: "High Risk"})
    
    fig = px.bar(
        counts,
        x="Risk Label",
        y="count",
        color="Risk Label",
        color_discrete_map={"Low/Medium Risk": "#4CAF50", "High Risk": "#F44336"},
        title="Patient Risk Classification Distribution",
        labels={"count": "Number of Patients", "Risk Label": "Patient Risk Status"},
        template="plotly_white"
    )
    fig.update_layout(title_font_size=18)
    return fig

def plot_correlation_heatmap(df, num_cols):
    """
    Generate a Plotly heatmap showing correlation among selected numerical features.
    """
    corr = df[num_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
        template="plotly_white"
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="",
        yaxis_title=""
    )
    return fig

def plot_feature_importance(model, feature_names=None, top_n=10):
    """
    Extract and plot feature importance from a trained model.
    """
    if hasattr(model, "steps"):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.steps[-1][1]
    else:
        preprocessor = None
        estimator = model
        
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        
        # Get dynamic feature names if preprocessor is available
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                pass
                
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
        # Clean up column transformer output prefixes (e.g. num__, cat__)
        clean_names = []
        for name in feature_names:
            if "__" in name:
                clean_names.append(name.split("__")[1])
            else:
                clean_names.append(name)
            
        feat_imp = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_n)
        
        fig = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title=f"Top {top_n} Feature Importances",
            template="plotly_white"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_font_size=18)
        return fig
    else:
        return None

def plot_cost_gauge(pred_value, max_val=150000):
    """
    Generate a Plotly gauge chart for the predicted medical cost.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Annual Cost ($)", 'font': {'size': 20}},
        number = {'prefix': "$", 'valueformat': ",.2f"},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5000], 'color': '#E8F5E9'},
                {'range': [5000, 15000], 'color': '#FFF9C4'},
                {'range': [15000, max_val], 'color': '#FFEBEE'}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_risk_meter(prob):
    """
    Generate a gauge chart for patient risk probability.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "High-Risk Probability (%)", 'font': {'size': 20}},
        number = {'suffix': "%", 'valueformat': ".1f"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#3F51B5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#C8E6C9'},    # Low risk Green
                {'range': [35, 70], 'color': '#FFE082'},   # Medium risk Orange
                {'range': [70, 100], 'color': '#FFCDD2'}   # High risk Red
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_waterfall_contribution(patient_data_dict, base_cost=2500):
    """
    Waterfall chart showing patient parameters impact on medical cost.
    """
    labels = ["Base Cost"]
    values = [base_cost]
    
    # 1. Age
    age = patient_data_dict.get("age", 40)
    age_contrib = (age ** 1.1) * 30.0 - (40 ** 1.1) * 30.0
    labels.append("Age Adjustment")
    values.append(age_contrib)
    
    # 2. BMI
    bmi = patient_data_dict.get("bmi", 25.0)
    bmi_contrib = (bmi - 25.0) * 180.0
    labels.append("BMI Impact")
    values.append(bmi_contrib)
    
    # 3. Smoking
    smoker = patient_data_dict.get("smoker", "Never")
    smoker_contrib = 14000.0 if smoker == "Current" else (3000.0 if smoker == "Former" else 0.0)
    labels.append("Smoker Impact")
    values.append(smoker_contrib)
    
    # 4. Chronic Diseases
    diabetes = patient_data_dict.get("diabetes", 0)
    hypertension = patient_data_dict.get("hypertension", 0)
    cardio = patient_data_dict.get("cardiovascular_disease", 0)
    kidney = patient_data_dict.get("kidney_disease", 0)
    
    chronic_contrib = (diabetes * 3500.0 + hypertension * 2000.0 + cardio * 8500.0 + kidney * 6500.0)
    labels.append("Chronic Conditions")
    values.append(chronic_contrib)
    
    # 5. Hospital Visits
    hosp_visits = patient_data_dict.get("visits_last_year", 0)
    days_hosp = patient_data_dict.get("days_hospitalized_last_3yrs", 0)
    util_contrib = hosp_visits * 1200.0 + days_hosp * 900.0
    labels.append("Hospital Utilization")
    values.append(util_contrib)
    
    total = sum(values)
    measures = ["relative"] * len(values)
    
    fig = go.Figure(go.Waterfall(
        name = "Contribution", 
        orientation = "v",
        measure = measures + ["total"],
        x = labels + ["Predicted Total"],
        textposition = "outside",
        text = [f"+${val:,.0f}" if val >= 0 else f"-${abs(val):,.0f}" for val in values] + [f"${total:,.0f}"],
        y = values + [total],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#4CAF50"}},
        increasing = {"marker":{"color":"#F44336"}},
        totals = {"marker":{"color":"#1E88E5"}}
    ))
    
    fig.update_layout(
        title = "Approximate Cost Contribution Analysis",
        showlegend = False,
        template = "plotly_white",
        title_font_size=18,
        height=400
    )
    return fig
