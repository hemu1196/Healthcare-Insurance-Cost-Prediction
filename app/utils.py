import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Corporate Colors Config
COLOR_PRIMARY = "#0F4C81"
COLOR_ACCENT = "#00A8E8"
COLOR_SUCCESS = "#2ECC71"
COLOR_WARNING = "#F39C12"
COLOR_DANGER = "#E74C3C"

def plot_cost_distribution(df):
    """
    Generate a Plotly histogram/KDE for the distribution of medical costs.
    """
    fig = px.histogram(
        df, 
        x="annual_medical_cost", 
        color_discrete_sequence=[COLOR_PRIMARY],
        marginal="box",
        title="Distribution of Annual Medical Costs",
        labels={"annual_medical_cost": "Annual Medical Cost ($)"},
        template="plotly_white"
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81",
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
        color_discrete_map={"Low/Medium Risk": COLOR_SUCCESS, "High Risk": COLOR_DANGER},
        title="Patient Risk Classification Distribution",
        labels={"count": "Number of Patients", "Risk Label": "Patient Risk Status"},
        template="plotly_white"
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
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
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81",
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
        
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                pass
                
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
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
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            font_family="Inter",
            title_font_family="Poppins",
            title_font_size=18,
            title_font_color="#0F4C81"
        )
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
        title = {'text': "Predicted Annual Cost ($)", 'font': {'size': 20, 'family': 'Poppins', 'color': COLOR_PRIMARY}},
        number = {'prefix': "$", 'valueformat': ",.2f", 'font': {'size': 32, 'family': 'Poppins'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': COLOR_PRIMARY},
            'bar': {'color': COLOR_PRIMARY},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5000], 'color': '#E8F8F5'},
                {'range': [5000, 15000], 'color': '#FEF9E7'},
                {'range': [15000, max_val], 'color': '#FDEDEC'}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), font_family="Inter")
    return fig

def plot_risk_meter(prob):
    """
    Generate a gauge chart for patient risk probability.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "High-Risk Probability (%)", 'font': {'size': 20, 'family': 'Poppins', 'color': COLOR_PRIMARY}},
        number = {'suffix': "%", 'valueformat': ".1f", 'font': {'size': 32, 'family': 'Poppins'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': COLOR_PRIMARY},
            'bar': {'color': COLOR_PRIMARY},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#E8F8F5'},    # Low risk (green)
                {'range': [35, 70], 'color': '#FEF9E7'},   # Medium risk (orange)
                {'range': [70, 100], 'color': '#FDEDEC'}   # High risk (red)
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), font_family="Inter")
    return fig

def plot_waterfall_contribution(patient_data_dict, base_cost=2500):
    """
    Waterfall chart showing patient parameters impact on medical cost.
    """
    labels = ["Base Cost"]
    values = [base_cost]
    
    age = patient_data_dict.get("age", 40)
    age_contrib = (age ** 1.1) * 30.0 - (40 ** 1.1) * 30.0
    labels.append("Age Adjustment")
    values.append(age_contrib)
    
    bmi = patient_data_dict.get("bmi", 25.0)
    bmi_contrib = (bmi - 25.0) * 180.0
    labels.append("BMI Impact")
    values.append(bmi_contrib)
    
    smoker = patient_data_dict.get("smoker", "Never")
    smoker_contrib = 14000.0 if smoker == "Current" else (3000.0 if smoker == "Former" else 0.0)
    labels.append("Smoker Impact")
    values.append(smoker_contrib)
    
    diabetes = patient_data_dict.get("diabetes", 0)
    hypertension = patient_data_dict.get("hypertension", 0)
    cardio = patient_data_dict.get("cardiovascular_disease", 0)
    kidney = patient_data_dict.get("kidney_disease", 0)
    
    chronic_contrib = (diabetes * 3500.0 + hypertension * 2000.0 + cardio * 8500.0 + kidney * 6500.0)
    labels.append("Chronic Conditions")
    values.append(chronic_contrib)
    
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
        decreasing = {"marker":{"color":COLOR_SUCCESS}},
        increasing = {"marker":{"color":COLOR_DANGER}},
        totals = {"marker":{"color":COLOR_PRIMARY}}
    ))
    
    fig.update_layout(
        title = "Approximate Cost Contribution Analysis",
        showlegend = False,
        template = "plotly_white",
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81",
        height=400
    )
    return fig

# --- NEW PREMIUM PLOTS FOR PERFORMANCE AND DATA ---

def plot_confusion_matrix(cm):
    """
    Generate an interactive Plotly heatmap for a confusion matrix.
    """
    x_labels = ['Low/Med Risk', 'High Risk']
    y_labels = ['Low/Med Risk', 'High Risk']
    
    fig = px.imshow(
        cm,
        text_auto=True,
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
        title="Confusion Matrix Diagnostics",
        template="plotly_white"
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Generate an interactive Plotly line chart representing the ROC Curve.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color=COLOR_PRIMARY, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81",
        legend=dict(x=0.6, y=0.15)
    )
    return fig

def plot_residual_analysis(y_true, y_pred):
    """
    Generate a scatter plot representing residuals vs predicted values.
    """
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        opacity=0.5,
        color_discrete_sequence=[COLOR_DANGER],
        labels={"x": "Predicted Value ($)", "y": "Residual ($)"},
        title="Residuals vs. Predicted Values Analysis",
        template="plotly_white"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Generate a scatter plot representing actual vs predicted values.
    """
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        opacity=0.5,
        color_discrete_sequence=[COLOR_PRIMARY],
        labels={"x": "Actual Value ($)", "y": "Predicted Value ($)"},
        title="Actual vs. Predicted Diagnostics",
        template="plotly_white"
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red", width=2),
        x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max()
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
    return fig

def plot_pca_clusters(X_scaled, labels):
    """
    Generate PCA cluster scatter plot.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Cohort ID"] = labels.astype(str)
    
    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="Cohort ID",
        opacity=0.6,
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING],
        title="Patient Cohort Clustering via PCA (2D Projection)",
        template="plotly_white"
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
    return fig

def plot_tsne_clusters(X_scaled, labels):
    """
    Generate t-SNE cluster scatter plot.
    To save computation speed, we run t-SNE on a subsample of 1000 points.
    """
    from sklearn.manifold import TSNE
    subsample_idx = np.random.choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
    X_sub = X_scaled[subsample_idx]
    labels_sub = labels[subsample_idx]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=300)
    X_tsne = tsne.fit_transform(X_sub)
    
    df_tsne = pd.DataFrame(X_tsne, columns=["t-SNE 1", "t-SNE 2"])
    df_tsne["Cohort ID"] = labels_sub.astype(str)
    
    fig = px.scatter(
        df_tsne,
        x="t-SNE 1",
        y="t-SNE 2",
        color="Cohort ID",
        opacity=0.7,
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING],
        title="Patient Cohort Clustering via t-SNE Visualization",
        template="plotly_white"
    )
    fig.update_layout(
        font_family="Inter",
        title_font_family="Poppins",
        title_font_size=18,
        title_font_color="#0F4C81"
    )
    return fig
