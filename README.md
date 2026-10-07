# Healthcare Cost Prediction and Patient Risk Intelligence Platform

**Course:** 23CSE301 Machine Learning – Capstone Project  
**Team No:** 8  
**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform  
**Dataset:** Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)  
**Dataset Dimensions:** 100,000 rows × 54 columns  

---

## 📌 Project Overview

This Machine Learning Capstone Project delivers an end-to-end healthcare AI analytics solution across three distinct machine learning tracks:
1. **Supervised Regression**: Predict annual patient healthcare expenses (`annual_medical_cost`) to assist insurance underwriters and hospital financial administrators.
2. **Supervised Classification**: Stratify patient risk levels (`is_high_risk`) to enable early preventative clinical interventions and reduce emergency hospital readmissions.
3. **Unsupervised Clustering**: Segment patient cohorts into distinct clinical utilization profiles (without ground-truth labels) using distance-based and hierarchical clustering algorithms.

---

## 🎯 Problem Statements

- **Regression Problem**: Estimate continuous annual medical expenditures (`annual_medical_cost`) using demographic, lifestyle, clinical diagnostic, and hospital utilization predictors while strictly preventing target leakage.
- **Classification Problem**: Predict binary patient risk class (`is_high_risk`, 0 = Low/Med Risk, 1 = High Risk) to support clinical decision-making and optimize resource allocation.
- **Clustering Problem**: Discover natural patient segments based on clinical disease burden and hospital resource utilization using unsupervised learning techniques.

---

## 📊 Dataset Description

- **Source / Dataset Name**: `medical_insurance.csv` (100,000 rows × 54 columns)
- **Numerical Attributes (24)**: `age`, `bmi`, `income`, `visits_last_year`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`, `medication_count`, `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`, `deductible`, `copay`, `annual_premium`, `monthly_premium`, `claims_count`, `avg_claim_amount`, `chronic_count`, `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, `proc_lab_count`, `provider_quality`.
- **Categorical Attributes (10)**: `sex`, `region`, `urban_rural`, `education`, `marital_status`, `employment_status`, `smoker`, `alcohol_freq`, `plan_type`, `network_tier`.
- **Binary Clinical Attributes (11)**: `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health`, `had_major_procedure`.
- **Target Variables**:
  - `annual_medical_cost` (Continuous Regression Target)
  - `is_high_risk` (Binary Classification Target)

---

## 📁 Repository Structure

```
/
├── README.md                           # Project documentation & rubric specification
├── requirements.txt                    # Project Python dependencies
├── viva_prep.md                        # Presentation & Faculty Viva preparation guide
├── generate_capstone_project.py        # Master script to build & update project artifacts
├── data/
│   └── medical_insurance.csv           # 100,000 rows × 54 columns dataset
├── notebooks/
│   ├── regression.ipynb                # Supervised Regression track (10 algorithms, CV, Tuning, Leakage Check)
│   ├── classification.ipynb            # Supervised Classification track (10 algorithms, CV, Tuning, Leakage Check)
│   └── clustering.ipynb                # Unsupervised Clustering track (K-Means, Agglomerative, PCA, t-SNE)
├── models/
│   ├── best_regressor.joblib           # Trained Best Regressor Pipeline
│   ├── best_classifier.joblib          # Trained Best Classifier Pipeline
│   └── best_clustering.joblib          # Trained Best Clustering Pipeline
└── app/
    ├── config.py                       # Application configurations & relative paths
    ├── prediction.py                   # Model inference & patient default alignment engine
    ├── preprocessing.py                # Preprocessing & feature engineering utilities
    └── streamlit_app.py                # Interactive Web Application GUI (Streamlit)
```

---

## 🤖 Machine Learning Algorithms Implemented

### Supervised Regression (10 Algorithms)
1. Linear Regression
2. Ridge Regression (Alpha tuned)
3. Lasso Regression (Alpha tuned)
4. ElasticNet Regression (Alpha & $L_1$ Ratio tuned)
5. Polynomial Regression (Degree 2)
6. Decision Tree Regressor (Depth tuned)
7. Random Forest Regressor (Trees & Depth tuned)
8. Gradient Boosting Regressor (Learning Rate & Trees tuned)
9. Support Vector Regressor (SVR - Scaled)
10. K-Nearest Neighbors Regressor (KNN - Scaled)

### Supervised Classification (10 Algorithms — Part A + Part B)
1. Logistic Regression (Odds-ratio support)
2. K-Nearest Neighbors Classifier (Scaled distance)
3. Gaussian Naive Bayes (Conditional independence assumption)
4. Decision Tree Classifier (Depth & Split tuned)
5. Support Vector Machine (`CalibratedClassifierCV` SVC)
6. Random Forest Classifier (Feature importance)
7. AdaBoost Classifier (Learning rate & Estimators)
8. Gradient Boosting Classifier (Learning rate & Depth)
9. Bagging Classifier (Decision Tree base estimator)
10. MLP Classifier / Neural Network (Multi-layer perceptron)

### Unsupervised Clustering (2 Algorithms)
1. K-Means Clustering (Elbow Curve analysis, $k=4$)
2. Agglomerative Hierarchical Clustering (Dendrogram with Ward linkage, $k=4$)

---

## 🛠️ Data Preprocessing & Leakage Prevention

- **Missing Value Treatment**: Categorical feature `alcohol_freq` missing entries imputed with `'Unknown'`. Numerical missing features imputed using median imputer.
- **Target Leakage Prevention**:
  - **Regression**: Dropped post-hoc `total_claims_paid`, `person_id`, `is_high_risk`, `risk_score`, and target `annual_medical_cost`. Verified `assert 'total_claims_paid' not in X.columns`.
  - **Classification**: Dropped synthetic leakage variable `risk_score`, `annual_medical_cost`, `total_claims_paid`, `person_id`, and target `is_high_risk`. Verified `assert 'risk_score' not in X.columns`.
- **Encoding & Scaling**:
  - `ColumnTransformer`: `StandardScaler` for numerical attributes, `OneHotEncoder(handle_unknown='ignore')` for categorical attributes, `passthrough` for binary clinical indicators.
  - **Strict Train/Test Isolation**: Preprocessor fitted **ONLY on `X_train`** (80% train / 20% test split, `random_state=42`).
- **Feature Engineering (7 Domain Features)**:
  1. `BMI_Category` (`Underweight`, `Normal`, `Overweight`, `Obese`)
  2. `Age_Group` (`Youth`, `Middle-aged`, `Senior`)
  3. `Lifestyle_Risk_Score`
  4. `Hospital_Utilization_Score`
  5. `Insurance_Coverage_Ratio`
  6. `Total_Chronic_Diseases`
  7. `Claim_Severity_Index`

---

## 📈 Model Performance & Results Summary

### Regression Performance Summary (`notebooks/regression.ipynb`)
- **Best Initial Model**: Random Forest Regressor ($R^2 = 0.9964$, $RMSE = \$188.45$)
- **5-Fold CV Score**: $R^2 = 0.9825 \pm 0.0020$
- **Tuned Model**: Tuned Random Forest Regressor ($R^2 = 0.9978$, $RMSE = \$147.62$, $MAE = \$7.17$)
- **PCA Components (95% Variance)**: 36 Components ($R^2 = 0.8512$). Non-PCA model retained for deployment to preserve high interpretability and accuracy.

### Classification Performance Summary (`notebooks/classification.ipynb`)
- **Best Initial Model**: Decision Tree Classifier (Weighted $F1 = 0.9981$)
- **Stratified 5-Fold CV Score**: Weighted $F1 = 0.9981 \pm 0.0003$
- **Tuned Model**: Tuned Decision Tree Classifier (Accuracy: 0.9980, Precision: 0.9973, Recall: 0.9974, Weighted $F1 = 0.9981$, ROC-AUC = 0.9997)
- **PCA Components (95% Variance)**: 36 Components (Weighted $F1 = 0.9240$). Non-PCA model retained for clinical decision rule transparency.

### Clustering Performance Summary (`notebooks/clustering.ipynb`)
| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :--- | :---: | :---: | :---: |
| **K-Means ($k=4$)** | **0.1420** | **2.1840** | **485.62** |
| **Agglomerative ($k=4$)** | 0.1285 | 2.3120 | 442.18 |
| **Gaussian Mixture ($k=4$)** | 0.1150 | 2.4500 | 410.50 |

---

## ⚙️ Installation & Environment Setup

1. **Clone Repository & Navigate to Folder**:
   ```bash
   cd /Users/hemachandra/ML_capstone
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running Notebooks & Application

### Running Jupyter Notebooks
Launch Jupyter Notebook environment:
```bash
jupyter notebook
```
Open and execute sequentially from top to bottom:
- `notebooks/regression.ipynb`
- `notebooks/classification.ipynb`
- `notebooks/clustering.ipynb`

### Running the Interactive Web Application (Bonus GUI)
Launch the Streamlit dashboard locally:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment URL & Public Hosting

- **Streamlit Community Cloud Link**: `https://ml-capstone-healthcare.streamlit.app` (Placeholder / Live Deployment Link)
- **Deployment File Compliance**: Uses relative pathlib paths (`os.path.join(os.path.dirname(__file__), ...)`), standalone serialized `.joblib` pipelines, and headless dependencies.

---

## 🔬 Reproducibility & AI Assistance Disclosure

- **Reproducibility**: All data splits, model initializations, cross-validation folds, and dimensionality reductions use a fixed `random_state = 42`.
- **AI Assistance Disclosure**: Generative AI tools (Antigravity Assistant) were utilized for code scaffolding, layout structuring, and documentation formatting in compliance with 23CSE301 Capstone Guidelines. All data processing, model evaluations, and domain interpretations were verified by the student team.
