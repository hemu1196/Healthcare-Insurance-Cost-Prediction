# Healthcare Cost Prediction and Patient Risk Intelligence Platform

## Academic Course Information
* **Course**: B.Tech. (Computer Science and Engineering) - III Year — 23CSE301 Machine Learning
* **Task Scope**: **Review 1** Complete (Full Regression Track + Classification Track Part A + Feature Engineering + GUI / Streamlit Implementation)

---

## 📋 Project Overview & Problem Statement
Predicting patient healthcare costs and clinical risks is critical for insurance underwriting, hospital resource planning, and proactive preventative care. 

This platform tackles this challenge using three machine learning tracks:
1. **Track 1: Cost Regression**: Predicts patient annual medical costs (`annual_medical_cost`) using demographic, lifestyle, and clinical predictors. We implement and evaluate 10 regression algorithms.
2. **Track 2: Risk Classification**: Identifies high-risk patients (`is_high_risk`) who are likely to incur major medical costs. We train and compare 5 classification algorithms for Part A.
3. **Track 3: Patient Segmentation**: Segments the patient cohort using KMeans clustering based on clinical metrics and resource utilization.

---

## 🗃 Dataset Description
We use the Kaggle Medical Insurance Cost Prediction dataset (downloaded locally at `/Users/hemachandra/Downloads/medical_insurance.csv`) consisting of **100,000 rows** and **54 columns**:
* **Demographics**: `age`, `sex` ("Female", "Male", "Other"), `region`, `urban_rural`, `income`, `education`, `marital_status`, `employment_status`, `household_size`, `dependents`.
* **Lifestyle**: `bmi`, `smoker` ("Never", "Current", "Former"), `alcohol_freq` ("None", "Occasional", "Weekly", "Daily").
* **Clinical History**: `diabetes`, `hypertension`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health`, `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`.
* **Healthcare Utilization**: `visits_last_year`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`, `medication_count`, `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, `proc_lab_count`, `had_major_procedure`.
* **Policy details**: `plan_type`, `network_tier`, `deductible`, `copay`, `policy_term_years`, `policy_changes_last_2yrs`, `provider_quality`, `annual_premium`, `monthly_premium`.
* **Claims**: `claims_count`, `avg_claim_amount`, `total_claims_paid`.
* **Targets**: `annual_medical_cost` (regression) and `is_high_risk` (classification).

---

## 🛠 Feature Engineering Summary
To enhance model performance, the following 7 clinical/policy features are engineered inside `app/preprocessing.py`:
1. **BMI Category**: Standard clinical bins (`Underweight`, `Normal`, `Overweight`, `Obese`).
2. **Age Group**: Generational grouping (`Youth`, `Middle-aged`, `Senior`).
3. **Lifestyle Risk Score**: Weighted score based on tobacco use, high alcohol consumption, high BMI, and mental health indicators.
4. **Hospital Utilization Score**: Cumulative weighted count of visits, hospitalizations, days hospitalized, surgeries, and imaging.
5. **Insurance Coverage Ratio**: Ratio showing the proportion of coverage provided after premium relative to deductible.
6. **Total Chronic Disease Count**: Integer sum of active chronic comorbidities.
7. **Claim Severity Index**: Diagnostic severity proxy combining age, obesity, chronic comorbidities, and hospital visits.

---

## 📊 Summary of Model Performance

### Track 1: Regression Results (Cost Prediction)
All 10 regression algorithms were evaluated using $R^2$, $RMSE$, and $MAE$ on the held-out test split:

| Model | $R^2$ Score | $RMSE$ ($) | $MAE$ ($) |
| :--- | :---: | :---: | :---: |
| **Tuned Random Forest** | **0.9978** | **147.62** | **7.17** |
| Gradient Boosting Regressor | 0.9969 | 173.62 | 88.12 |
| Random Forest Regressor | 0.9964 | 188.45 | 76.51 |
| Tuned Gradient Boosting | 0.9963 | 189.76 | 87.37 |
| Decision Tree Regressor | 0.9927 | 267.59 | 136.81 |
| Polynomial Regression | 0.9665 | 574.39 | 319.41 |
| Lasso Regression | 0.9662 | 576.30 | 310.23 |
| Ridge Regression | 0.9662 | 576.63 | 312.86 |
| Linear Regression | 0.9662 | 576.65 | 312.87 |
| ElasticNet | 0.9172 | 902.65 | 524.57 |
| KNN Regressor | 0.8002 | 1,402.00 | 917.33 |
| Support Vector Regressor (SVR) | 0.7530 | 1,559.03 | 391.73 |

* **5-Fold Cross Validation R² (on top 2 models)**:
  * Tuned Random Forest: **0.9954**
  * Tuned Gradient Boosting: **0.9942**

### Track 2: Classification Results (Patient Risk Part A)
Evaluation metrics for the 5 classification algorithms:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned Decision Tree** | **0.9988** | **0.9986** | **0.9981** | **0.9988** | **0.9987** |
| Tuned Logistic Regression | 0.9987 | 0.9978 | 0.9986 | 0.9987 | 0.9998 |
| Decision Tree | 0.9977 | 0.9975 | 0.9962 | 0.9977 | 0.9989 |
| Logistic Regression | 0.9954 | 0.9973 | 0.9902 | 0.9954 | 0.9998 |
| Support Vector Machine (SVC) | 0.9656 | 0.9675 | 0.9380 | 0.9655 | 0.9962 |
| K-Nearest Neighbors (KNN) | 0.8586 | 0.9133 | 0.6801 | 0.8531 | 0.9399 |
| Naive Bayes (Gaussian) | 0.8355 | 0.7816 | 0.7671 | 0.8352 | 0.9105 |

---

## 📁 Repository Structure
```
ML_capstone/
├── README.md                 # Project README report
├── requirements.txt          # Python dependencies
├── generate_notebooks.py     # Notebooks builder script
├── train_clustering.py       # Extra clustering training script
├── data/
│   └── insurance_extended.csv # Locally copied Kaggle dataset (100k rows x 54 cols)
├── models/
│   ├── best_regressor.joblib  # Saved Tuned Random Forest model
│   ├── best_classifier.joblib # Saved Tuned Decision Tree model
│   └── best_clustering.joblib # Saved KMeans clustering model
├── results/
│   ├── regression_metrics.csv # Metrics log for regressors
│   ├── classification_metrics.csv # Metrics log for classifiers
│   └── clustering_metrics.csv # Metrics log for clustering
├── notebooks/
│   ├── regression.ipynb       # Fully-run regression notebook
│   └── classification.ipynb   # Fully-run classification notebook
└── app/
    ├── config.py              # Configuration & hyperparameters
    ├── preprocessing.py       # Feature engineering & scaling pipelines
    ├── model_training.py      # Pipelines for training & tuning
    ├── prediction.py          # Predictions inference wrapper
    ├── utils.py               # Plotly visualisations and metric utilities
    └── streamlit_app.py       # Streamlit UI dashboard code
```

---

## 🚀 Setup and How-to-Run Instructions

### Prerequisites
Make sure Python 3.10+ is installed on your machine.

### 1. Set Up Virtual Environment & Install Dependencies
```bash
cd /Users/hemachandra/Documents/ML_capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Model Training (Notebooks / Scripts)
The notebooks are already pre-run and outputs are visible. However, you can re-run them or run the scripts using:
```bash
python3 -c "import app.preprocessing as prep; import app.model_training as mt; X_tr, X_te, y_tr, y_te, c, n, b = prep.prepare_data_regression(); mt.train_regression(X_tr, X_te, y_tr, y_te, prep.get_preprocessor(c, n, b))"
```

### 3. Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
This will start the server locally and open the platform in your browser at `http://localhost:8501`.
