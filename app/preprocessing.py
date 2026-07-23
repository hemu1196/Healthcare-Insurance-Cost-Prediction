import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import app.config as config

def engineer_features(df_input):
    """
    Apply feature engineering to the raw input dataframe.
    """
    df = df_input.copy()
    
    # Fill missing values in alcohol_freq (nan implies no regular consumption)
    if "alcohol_freq" in df.columns:
        df["alcohol_freq"] = df["alcohol_freq"].fillna("None")
    
    # 1. BMI Category
    if "bmi" in df.columns:
        conditions_bmi = [
            (df["bmi"] < 18.5),
            (df["bmi"] >= 18.5) & (df["bmi"] < 25),
            (df["bmi"] >= 25) & (df["bmi"] < 30),
            (df["bmi"] >= 30)
        ]
        choices_bmi = ["Underweight", "Normal", "Overweight", "Obese"]
        df["BMI_Category"] = np.select(conditions_bmi, choices_bmi, default="Normal")
    else:
        df["BMI_Category"] = "Normal"
    
    # 2. Age Group
    if "age" in df.columns:
        conditions_age = [
            (df["age"] < 30),
            (df["age"] >= 30) & (df["age"] <= 55),
            (df["age"] > 55)
        ]
        choices_age = ["Youth", "Middle-aged", "Senior"]
        df["Age_Group"] = np.select(conditions_age, choices_age, default="Middle-aged")
    else:
        df["Age_Group"] = "Middle-aged"
    
    # 3. Lifestyle Risk Score
    smoker_weight = 0.0
    if "smoker" in df.columns:
        smoker_weight = np.where(df["smoker"] == "Current", 3.0, np.where(df["smoker"] == "Former", 1.0, 0.0))
        
    alcohol_weight = 0.0
    if "alcohol_freq" in df.columns:
        alcohol_map = {"Daily": 1.5, "Weekly": 0.75, "Occasional": 0.25, "None": 0.0}
        alcohol_weight = df["alcohol_freq"].map(alcohol_map).fillna(0.0)
        
    bmi_weight = 0.0
    if "bmi" in df.columns:
        bmi_weight = np.where(df["bmi"] >= 30.0, 1.0, 0.0)
        
    mental_weight = 0.0
    if "mental_health" in df.columns:
        mental_weight = df["mental_health"] * 0.5
        
    df["Lifestyle_Risk_Score"] = smoker_weight + alcohol_weight + bmi_weight + mental_weight
    
    # 4. Hospital Utilization Score
    visits = df["visits_last_year"] if "visits_last_year" in df.columns else 0.0
    hosp_3yrs = df["hospitalizations_last_3yrs"] if "hospitalizations_last_3yrs" in df.columns else 0.0
    days_hosp = df["days_hospitalized_last_3yrs"] if "days_hospitalized_last_3yrs" in df.columns else 0.0
    surgery = df["proc_surgery_count"] if "proc_surgery_count" in df.columns else 0.0
    imaging = df["proc_imaging_count"] if "proc_imaging_count" in df.columns else 0.0
    
    df["Hospital_Utilization_Score"] = (
        visits * 1.0 + 
        hosp_3yrs * 3.0 + 
        days_hosp * 0.5 + 
        surgery * 2.0 + 
        imaging * 1.0
    )
    
    # 5. Insurance Coverage Ratio
    premium = df["annual_premium"] if "annual_premium" in df.columns else 1000.0
    ded = df["deductible"] if "deductible" in df.columns else 500.0
    df["Insurance_Coverage_Ratio"] = premium / (ded + premium + 1e-5)
    df["Insurance_Coverage_Ratio"] = np.clip(df["Insurance_Coverage_Ratio"], 0.0, 1.0)
    
    # 6. Total Chronic Diseases
    if "chronic_count" in df.columns:
        df["Total_Chronic_Diseases"] = df["chronic_count"]
    else:
        chronic_cols = ["hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease", "cancer_history", "kidney_disease", "liver_disease", "arthritis"]
        available_cols = [c for c in chronic_cols if c in df.columns]
        df["Total_Chronic_Diseases"] = df[available_cols].sum(axis=1) if available_cols else 0.0
    
    # 7. Claim Severity Index
    major_proc = df["had_major_procedure"] if "had_major_procedure" in df.columns else 0.0
    df["Claim_Severity_Index"] = (
        (df["age"] * 0.05) + 
        (df["bmi"] * 0.1) + 
        (df["Total_Chronic_Diseases"] * 1.5) + 
        (df["Hospital_Utilization_Score"] * 0.5) +
        major_proc * 1.5
    )
    
    return df

def get_preprocessor(categorical_features, numerical_features, binary_features):
    """
    Creates the ColumnTransformer preprocessor for scaling and encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
            ("bin", "passthrough", binary_features)
        ]
    )
    return preprocessor

def prepare_data_regression(filepath=config.DATA_PATH):
    """
    Load dataset, engineer features, split into train and test sets, and return X and y.
    """
    df = pd.read_csv(filepath)
    df = engineer_features(df)
    
    # Drop identifiers and columns that lead to leakage
    # risk_score contains the target information. Drop it.
    # also drop person_id
    drop_cols = [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET, "person_id", "risk_score"]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[config.REGRESSION_TARGET]
    
    cat_cols = config.CATEGORICAL_FEATURES + ["BMI_Category", "Age_Group"]
    num_cols = config.NUMERICAL_FEATURES + [
        "Lifestyle_Risk_Score", "Hospital_Utilization_Score", 
        "Insurance_Coverage_Ratio", "Total_Chronic_Diseases", "Claim_Severity_Index"
    ]
    bin_cols = config.BINARY_CLINICAL_FEATURES
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols

def prepare_data_classification(filepath=config.DATA_PATH):
    """
    Load dataset, engineer features, split into train and test sets for classification, and return X and y.
    Note: risk_score is excluded to prevent leakage.
    """
    df = pd.read_csv(filepath)
    df = engineer_features(df)
    
    # Drop regression target (annual_medical_cost) to avoid leaking cost to risk, and drop is_high_risk.
    # Also drop person_id and risk_score
    drop_cols = [config.CLASSIFICATION_TARGET, config.REGRESSION_TARGET, "person_id", "risk_score"]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[config.CLASSIFICATION_TARGET]
    
    cat_cols = config.CATEGORICAL_FEATURES + ["BMI_Category", "Age_Group"]
    num_cols = config.NUMERICAL_FEATURES + [
        "Lifestyle_Risk_Score", "Hospital_Utilization_Score", 
        "Insurance_Coverage_Ratio", "Total_Chronic_Diseases", "Claim_Severity_Index"
    ]
    bin_cols = config.BINARY_CLINICAL_FEATURES
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols
