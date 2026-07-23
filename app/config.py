import os

# Base Directories
BASE_DIR = "/Users/hemachandra/Documents/ML_capstone"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# File Paths
DATA_PATH = os.path.join(DATA_DIR, "insurance_extended.csv")
BEST_REGRESSOR_PATH = os.path.join(MODEL_DIR, "best_regressor.joblib")
BEST_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_classifier.joblib")
REGRESSION_METRICS_PATH = os.path.join(RESULTS_DIR, "regression_metrics.csv")
CLASSIFICATION_METRICS_PATH = os.path.join(RESULTS_DIR, "classification_metrics.csv")

# Targets
REGRESSION_TARGET = "annual_medical_cost"
CLASSIFICATION_TARGET = "is_high_risk"

# Features Configuration
CATEGORICAL_FEATURES = [
    "sex", "region", "urban_rural", "education", "marital_status", 
    "employment_status", "smoker", "alcohol_freq", "plan_type", "network_tier"
]

NUMERICAL_FEATURES = [
    "age", "income", "household_size", "dependents", "bmi", "visits_last_year", 
    "hospitalizations_last_3yrs", "days_hospitalized_last_3yrs", "medication_count", 
    "systolic_bp", "diastolic_bp", "ldl", "hba1c", "deductible", "copay", 
    "policy_term_years", "policy_changes_last_2yrs", "provider_quality", 
    "annual_premium", "monthly_premium", "claims_count", "avg_claim_amount", 
    "total_claims_paid", "chronic_count", "proc_imaging_count", 
    "proc_surgery_count", "proc_physio_count", "proc_consult_count", "proc_lab_count"
]

BINARY_CLINICAL_FEATURES = [
    "hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease", 
    "cancer_history", "kidney_disease", "liver_disease", "arthritis", 
    "mental_health", "had_major_procedure"
]

ALL_PREDICTORS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_CLINICAL_FEATURES

# Engineered Features
ENGINEERED_FEATURES = [
    "BMI_Category", "Age_Group", "Lifestyle_Risk_Score", 
    "Hospital_Utilization_Score", "Insurance_Coverage_Ratio", 
    "Total_Chronic_Diseases", "Claim_Severity_Index"
]

# Random State
RANDOM_STATE = 42

# Hyperparameter Tuning Grids
REG_TUNING_GRIDS = {
    "Random Forest": {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [10, 15, None],
        "regressor__min_samples_split": [2, 5]
    },
    "Gradient Boosting": {
        "regressor__n_estimators": [50, 100],
        "regressor__learning_rate": [0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 5]
    }
}

CLS_TUNING_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"]
    },
    "Decision Tree": {
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10]
    }
}

