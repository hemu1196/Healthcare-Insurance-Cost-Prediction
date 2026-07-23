import os
import joblib
import pandas as pd
import numpy as np
import app.config as config
import app.preprocessing as preprocessing

_regressor = None
_classifier = None

def load_models():
    """
    Load the regression and classification models into memory if they aren't loaded yet.
    """
    global _regressor, _classifier
    if _regressor is None:
        if os.path.exists(config.BEST_REGRESSOR_PATH):
            _regressor = joblib.load(config.BEST_REGRESSOR_PATH)
        else:
            raise FileNotFoundError(f"Regressor model file not found at {config.BEST_REGRESSOR_PATH}. Run training first.")
    
    if _classifier is None:
        if os.path.exists(config.BEST_CLASSIFIER_PATH):
            _classifier = joblib.load(config.BEST_CLASSIFIER_PATH)
        else:
            raise FileNotFoundError(f"Classifier model file not found at {config.BEST_CLASSIFIER_PATH}. Run training first.")
            
    return _regressor, _classifier

def predict_cost(patient_data_dict):
    """
    Predict the annual medical cost for a given patient.
    Input should be a dictionary with keys matching the raw dataset columns.
    Returns: float (predicted cost)
    """
    regressor, _ = load_models()
    
    df_raw = pd.DataFrame([patient_data_dict])
    
    # Feature engineering
    df_engineered = preprocessing.engineer_features(df_raw)
    
    # Drop target columns to match model training input columns
    drop_cols = [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET, "person_id", "risk_score"]
    X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns], errors="ignore")
    
    # Predict
    pred = regressor.predict(X)[0]
    return float(pred)

def predict_risk(patient_data_dict):
    """
    Predict the risk level and high risk probability for a given patient.
    Input should be a dictionary with keys matching the raw dataset columns.
    Returns: dict with 'probability', 'risk_level', 'recommendations'
    """
    _, classifier = load_models()
    
    df_raw = pd.DataFrame([patient_data_dict])
    
    # Feature engineering
    df_engineered = preprocessing.engineer_features(df_raw)
    
    # Drop target columns
    drop_cols = [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET, "person_id", "risk_score"]
    X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns], errors="ignore")
    
    # Predict probability
    if hasattr(classifier, "predict_proba"):
        prob = classifier.predict_proba(X)[0][1]
    else:
        decision_val = classifier.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-decision_val))
        
    prob = float(prob)
    
    if prob >= 0.70:
        risk_level = "High"
    elif prob >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # Generate personalized recommendations based on actual dataset column names
    recs = []
    
    if patient_data_dict.get("smoker") == "Current":
        recs.append("Enroll in a structured Smoking Cessation Program to reduce cardiovascular and respiratory risks.")
    
    if patient_data_dict.get("bmi", 0) >= 30:
        recs.append("Schedule a session with a Registered Dietitian for weight management and metabolic health optimization.")
        
    if patient_data_dict.get("diabetes") == 1:
        recs.append("Monitor HbA1c levels every 3 months and verify adherence to prescribed anti-glycemic regimens.")
        
    if patient_data_dict.get("hypertension") == 1:
        recs.append("Check blood pressure twice daily. Aim for a target below 130/80 mmHg via sodium restriction and prescription compliance.")
        
    if patient_data_dict.get("cardiovascular_disease") == 1:
        recs.append("Routine cardiological review required every 6 months. Maintain low-dose aspirin or beta-blockers as directed.")
        
    if patient_data_dict.get("visits_last_year", 0) > 2:
        recs.append("Set up a telehealth primary care appointment to review the patient's care coordination plan and reduce emergency admissions.")
        
    if not recs:
        recs.append("Maintain current healthy lifestyle choices. Continue annual wellness checkups and preventative screenings.")
        
    return {
        "probability": prob,
        "risk_level": risk_level,
        "recommendations": recs
    }
