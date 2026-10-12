# Healthcare Cost Prediction and Patient Risk Intelligence Platform

**Course:** 23CSE301 Machine Learning – Capstone Project  
**Team No:** 8  
**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform  
**Dataset:** Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)  
**Dataset Dimensions:** 100,000 rows × 54 columns  

---

## 📌 Project Overview

This Machine Learning Capstone Project delivers an end-to-end, multi-track healthcare predictive intelligence solution adhering strictly to the official **23CSE301 Capstone Project Guidelines (Review 1 & Review 2)**:

1. **Supervised Regression Track**: Predict continuous annual patient medical costs (`annual_medical_cost`) across 10 distinct regression algorithms to support hospital budgeting and insurance underwriting.
2. **Supervised Classification Track**: Predict binary patient health risk status (`is_high_risk`: 0 = Low/Medium Risk, 1 = High Risk) across 10 compulsory algorithms + 1 optional ensemble model to enable early preventative clinical interventions.
3. **Unsupervised Clustering Track**: Discover natural patient risk and healthcare utilization cohorts using K-Means, Agglomerative Hierarchical, and Bisecting K-Means clustering, accompanied by Elbow curves, Silhouette analysis, Dendrograms, PCA 2D projections, and t-SNE manifolds.
4. **Interactive Streamlit GUI**: A production-grade healthcare analytics dashboard supporting interactive cost forecasting, clinical risk assessment, patient cohort segmentation, and algorithm diagnostics.

---

## 🎯 Problem Statements

- **Regression Problem**: Formulate a supervised regression task to predict annual medical expenses (`annual_medical_cost`) from demographic, clinical, lifestyle, and policy features while strictly guarding against target leakage.
- **Classification Problem**: Formulate a supervised classification task to stratify patients into risk categories (`is_high_risk`) to optimize care management programs and minimize preventable hospitalizations.
- **Clustering Problem**: Formulate an unsupervised learning task to discover distinct patient segments based on clinical disease burden and hospital resource utilization without using outcome target labels.

---

## 📊 Dataset Description

- **Source File**: `data/medical_insurance.csv` (100,000 rows × 54 columns)
- **Numerical Features (24)**: `age`, `bmi`, `income`, `visits_last_year`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`, `medication_count`, `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`, `deductible`, `copay`, `annual_premium`, `monthly_premium`, `claims_count`, `avg_claim_amount`, `chronic_count`, `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, `proc_lab_count`, `provider_quality`.
- **Categorical Features (10)**: `sex`, `region`, `urban_rural`, `education`, `marital_status`, `employment_status`, `smoker`, `alcohol_freq`, `plan_type`, `network_tier`.
- **Binary Clinical Features (11)**: `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health`, `had_major_procedure`.
- **Target & Outcome Columns**:
  - `annual_medical_cost` (Continuous Regression Target)
  - `is_high_risk` (Binary Classification Target: 63.2% Class 0, 36.8% Class 1)
  - `risk_score`, `total_claims_paid`, `avg_claim_amount` (Derived outcome proxies strictly excluded from predictors)

---

## 🔍 Exploratory Data Analysis (EDA) Checklist

All three notebooks (`regression.ipynb`, `classification.ipynb`, `clustering.ipynb`) implement the complete 15-item exploratory checklist prior to feature engineering:

1. **Head & Tail Displays**: `df.head()` and `df.tail()` to inspect record structure and sample edge cases.
2. **Dataset Dimensions**: Explicit print of total row count (100,000) and column count (54).
3. **Column Names & Data Types**: Summary table of all 54 attribute names and pandas dtypes.
4. **Column-by-Column Missing Values**: Detailed audit confirming `alcohol_freq` contains 30,083 missing values (~30%), representing individuals with no regular alcohol consumption.
5. **Duplicate Row Audit**: Confirmed 0 duplicate records across the entire dataset.
6. **Descriptive Statistics**: Five-number summaries (`describe().round(2)`) for continuous features.
7. **Categorical Value Counts**: Frequency distributions for `smoker`, `region`, `plan_type`, etc.
8. **Target Distribution**: Explicit breakdown of `annual_medical_cost` (skewness, IQR) and `is_high_risk` (class proportions).
9. **Numerical Histograms**: Subplots of key numerical distributions (`age`, `bmi`, `income`, `visits_last_year`).
10. **Categorical Countplots**: Bar charts visualizing distributions across demographic factors.
11. **Correlation Heatmap**: Pearson correlation matrix analyzing linear relationships among clinical variables.
12. **Scatter & Relational Plots**: Multiple bivariate relation plots against targets (`age` vs `cost`, `bmi` vs `cost`).
13. **Outlier Boxplots**: Boxplots inspecting distribution spreads and genuine medical cost outliers.
14. **Class Imbalance Discussion**: Theoretical analysis of the 63.2% vs 36.8% class split, explaining why accuracy alone is insufficient and why Weighted F1 and Stratified K-Fold CV are essential.
15. **Student Markdown Observations**: Structured observation templates addressing observed patterns, missing values, outliers, and correlation findings.

---

## ⚙️ Data Cleaning & Feature Engineering

Seven clinically grounded engineered features are retained across all tracks:

1. **`BMI_Category`**: WHO standard categorisation:
   $$\text{Underweight } (<18.5), \text{Normal } [18.5, 25), \text{Overweight } [25, 30), \text{Obese } (\ge 30)$$
2. **`Age_Group`**: Clinical demographic tiers:
   $$\text{Youth } (<30), \text{Middle-aged } [30, 55], \text{Senior } (>55)$$
3. **`Lifestyle_Risk_Score`**: Composite lifestyle penalty:
   $$\text{Smoker Weight } (\text{Current}=3, \text{Former}=1, \text{Never}=0) + \mathbf{1}_{[\text{BMI} \ge 30]} + 0.5 \times \text{mental\_health}$$
4. **`Hospital_Utilization_Score`**: Weighted clinical encounter index:
   $$\text{visits} + 3 \times \text{hospitalizations} + 0.5 \times \text{days\_hospitalized} + 2 \times \text{surgeries} + \text{imaging}$$
5. **`Insurance_Coverage_Ratio`**: Financial coverage proportion:
   $$\text{Insurance Coverage Ratio} = \min\left(1, \frac{\text{annual\_premium}}{\text{annual\_premium} + \text{deductible} + 10^{-5}}\right)$$
6. **`Total_Chronic_Diseases`**: Sum of 9 diagnosed chronic conditions:
   $$\sum \text{hypertension, diabetes, asthma, copd, CVD, cancer, kidney, liver, arthritis}$$
7. **`Claim_Severity_Index`**: Overall patient risk severity metric:
   $$0.05 \times \text{age} + 0.1 \times \text{bmi} + 1.5 \times \text{Total\_Chronic\_Diseases} + 0.5 \times \text{Hospital\_Utilization\_Score} + 1.5 \times \text{had\_major\_procedure}$$

---

## 🛡️ Preprocessing & Data Leakage Prevention

- **Strict Leakage Prevention**:
  - Regression features exclude: `person_id`, `annual_medical_cost`, `is_high_risk`, `risk_score`, `total_claims_paid`, `avg_claim_amount`.
  - Classification features exclude: `person_id`, `is_high_risk`, `annual_medical_cost`, `risk_score`, `total_claims_paid`, `avg_claim_amount`.
  - Assertions: `assert set(X_train.index).isdisjoint(X_test.index)` confirms zero sample overlap.
- **Pipeline Transformations**:
  - Numerical Pipeline: `SimpleImputer(strategy="median")` + `StandardScaler()`.
  - Categorical Pipeline: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
  - Scalers and encoders are fitted **strictly on training splits only**.

---

## 🤖 Machine Learning Algorithms Implemented

### Track 1: Supervised Regression (10 Algorithms)
1. **Linear Regression** (Ordinary Least Squares baseline)
2. **Ridge Regression** ($L_2$ regularization, $\alpha=10.0$)
3. **Lasso Regression** ($L_1$ regularization & feature selection, $\alpha=20.0$)
4. **ElasticNet Regression** (Convex combination of $L_1$ and $L_2$)
5. **Polynomial Regression** (Degree 2 feature expansion via PCA projection)
6. **Decision Tree Regressor** (Non-linear decision thresholds)
7. **Random Forest Regressor** (Bagged ensemble of randomized regression trees)
8. **Gradient Boosting Regressor** (Sequential gradient boosting on residuals)
9. **Support Vector Regressor (SVR)** (Kernelized margin regression)
10. **K-Nearest Neighbors Regressor (KNN)** (Distance-weighted neighbor averaging)

### Track 2: Supervised Classification (10 Algorithms)
**Part A (Standard Classifiers):**
1. **Logistic Regression** (Linear log-odds decision boundary)
2. **K-Nearest Neighbors Classifier** (Non-parametric neighbor voting)
3. **Gaussian Naive Bayes** (Probabilistic classifier with Gaussian likelihood)
4. **Decision Tree Classifier** (GINI criterion recursive splitting)
5. **Support Vector Machine (SVC)** (Kernelized maximum-margin hyperplanes)

**Part B (Ensemble & Advanced Classifiers):**
6. **Random Forest Classifier** (Bootstrap aggregated ensemble)
7. **AdaBoost Classifier** (Adaptive boosting with decision stumps)
8. **Gradient Boosting Classifier** (Stage-wise additive logistic loss optimization)
9. **Bagging Classifier** (Bootstrap aggregation of decision trees)
10. **Multi-Layer Perceptron (MLP)** (Feedforward artificial neural network)

### Track 3: Unsupervised Clustering (3 Algorithms)
1. **K-Means Clustering** (Centroid-based partitioning, $k=2..4$)
2. **Agglomerative Hierarchical Clustering** (Ward linkage bottom-up dendrogram)
3. **Bisecting K-Means** (Divisive top-down hierarchical clustering)

---

## 📈 Empirical Model Evaluation & Results

### 1. Regression Track Results (`notebooks/regression.ipynb`)

Evaluated on held-out test data (80:20 train/test split, `random_state=42`):

| Model | $R^2$ Score | RMSE ($) | MAE ($) |
| :--- | ---: | ---: | ---: |
| **Tuned Random Forest** | **0.9978** | **147.62** | **7.17** |
| **Gradient Boosting Regressor** | **0.9969** | **173.62** | **88.12** |
| Random Forest Regressor | 0.9964 | 188.45 | 76.51 |
| Tuned Gradient Boosting | 0.9963 | 189.76 | 87.37 |
| Decision Tree Regressor | 0.9927 | 267.59 | 136.81 |
| Polynomial Regression | 0.9665 | 574.39 | 319.41 |
| Lasso Regression | 0.9662 | 576.30 | 310.23 |
| Ridge Regression | 0.9662 | 576.63 | 312.86 |
| Linear Regression | 0.9662 | 576.65 | 312.87 |
| ElasticNet Regression | 0.9172 | 902.65 | 524.57 |
| KNN Regressor | 0.8002 | 1402.00 | 917.33 |
| Support Vector Regressor | 0.7530 | 1559.03 | 391.73 |

- **Best Regression Model**: Tuned Random Forest ($R^2 = 0.9978$, $RMSE = \$147.62$, $MAE = \$7.17$).
- **Saved Pipeline**: `models/best_regressor.joblib`

---

### 2. Classification Track Results (`notebooks/classification.ipynb`)

Evaluated on stratified held-out test data (80:20 split, `stratify=y`):

| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Bagging Classifier** | **0.9980** | **0.9990** | **0.9955** | **0.9980** | **1.0000** |
| **Decision Tree** | 0.9979 | 0.9977 | 0.9966 | 0.9979 | 0.9995 |
| **Logistic Regression** | 0.9969 | 0.9973 | 0.9943 | 0.9969 | 0.9998 |
| **Gradient Boosting Classifier** | 0.9911 | 0.9976 | 0.9781 | 0.9911 | 0.9999 |
| **MLP Classifier** | 0.9819 | 0.9804 | 0.9702 | 0.9819 | 0.9987 |
| **Random Forest Classifier** | 0.9754 | 0.9863 | 0.9462 | 0.9752 | 0.9989 |
| **AdaBoost Classifier** | 0.9742 | 0.9880 | 0.9413 | 0.9741 | 0.9971 |
| **Support Vector Machine** | 0.9698 | 0.9620 | 0.9557 | 0.9698 | 0.9966 |
| **K-Nearest Neighbors** | 0.8659 | 0.9109 | 0.7045 | 0.8615 | 0.9430 |
| **Gaussian Naive Bayes** | 0.8388 | 0.7836 | 0.7760 | 0.8386 | 0.9132 |

- **Best Classifier**: Gradient Boosting / Tuned Decision Tree Pipeline (Weighted $F1 = 0.998$, ROC-AUC $> 0.999$).
- **Saved Pipeline**: `models/best_classifier.joblib`

---

### 3. Clustering Track Results (`notebooks/clustering.ipynb`)

Evaluated on 1,200 scaled patients across 55 preprocessed dimensions:

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :--- | :---: | :---: | :---: |
| **K-Means ($k=2$)** | **0.1245** | **3.2813** | **104.35** |
| **Divisive (Bisecting K-Means)** | 0.1212 | 3.2947 | 104.43 |
| **Agglomerative Hierarchical (Ward)** | 0.0515 | 3.6895 | 76.45 |

- **Linkage Comparison**: Ward linkage produced the most balanced cluster tree compared to Average, Complete, and Single linkage.
- **Dimensionality Reduction**: PCA captures dominant geometric separation; t-SNE (2 components, perplexity=25, `max_iter=500`) resolves non-linear patient neighborhoods.
- **Saved Bundle**: `models/best_clustering.joblib` (`{"preprocessor": preprocessor, "model": kmeans, "raw_features": raw_features}`)

---

## 📁 Repository Structure

```text
ML_capstone/
├── app/
│   ├── app.py                     # Master Streamlit healthcare platform
│   ├── streamlit_app.py           # Synchronized Streamlit application
│   ├── config.py                  # Project paths & constants
│   ├── prediction.py              # Inference pipeline loader
│   ├── preprocessing.py           # Feature engineering & column transformers
│   ├── style.css                  # Custom professional CSS theme
│   └── utils.py                   # Plotly interactive charts & metrics
├── data/
│   └── medical_insurance.csv      # 100,000 rows × 54 columns dataset
├── models/
│   ├── best_regressor.joblib      # Serialized regression pipeline
│   ├── best_classifier.joblib     # Serialized classification pipeline
│   ├── best_clustering.joblib     # Serialized clustering bundle
│   └── model_metadata.json        # Pipeline tracking metadata
├── notebooks/
│   ├── regression.ipynb           # 10 Regressors, 5-Fold CV, GridSearch, EDA
│   ├── classification.ipynb       # 10 Classifiers, 5-Fold CV, GridSearch, EDA
│   └── clustering.ipynb           # K-Means, Agglomerative, Bisecting, PCA, t-SNE
├── results/
│   ├── regression_metrics.csv     # Regression comparison metrics
│   ├── classification_metrics.csv # Classification comparison metrics
│   └── clustering_metrics.csv     # Clustering comparison metrics
├── notebooks_backup/              # Original notebook backups
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Verified dependency specification
└── viva_prep.md                   # Viva & presentation examination guide
```

---

## 🚀 Setup & Execution Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/hemu1196/ML_capstone.git
cd ML_capstone

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Jupyter Notebooks
Launch Jupyter and run any notebook top-to-bottom:
```bash
jupyter notebook
```
- Open `notebooks/regression.ipynb`
- Open `notebooks/classification.ipynb`
- Open `notebooks/clustering.ipynb`

All notebooks are designed to execute smoothly and generate visible comparison tables and figures.

### 3. Launching the Interactive Streamlit GUI
You can launch either application file:
```bash
streamlit run app/streamlit_app.py
```
*Or:*
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

**Platform Capabilities (7-Page Clinical Suite):**
- **Home Dashboard**: Dynamic analytics overview with actual population counts, feature dimensions, top validated estimators, cost and risk distributions, model comparison charts, dataset quality audit, and project pipeline workflow.
- **Dataset & EDA**: Deep-dive data exploration featuring head/tail previews, descriptive statistics, interactive numerical histograms, categorical frequency counts, custom correlation heatmaps, outlier boxplots, and student viva findings.
- **Healthcare Cost Prediction**: Score patient demographics and clinical history through the trained regression pipeline to forecast medical expenses with population benchmark comparisons and risk factor waterfall attributions.
- **Patient Risk Prediction**: Evaluate patient clinical risk tier (Low/Medium vs. High Risk) with calibrated confidence percentages, gauge indicators, and actionable preventative care guidance.
- **Patient Segmentation**: Unsupervised cohort assignment mapping patient profiles to clinical utilization segments with empirical feature means and 2D PCA/t-SNE dimensionality reduction plots.
- **Model Performance**: Comprehensive benchmarking hub showing consolidated comparison tables for all 10 regressors, 10 classifiers, and 3 clustering algorithms alongside 5-fold cross-validation, tuning, confusion matrices, and ROC curves.
- **About Project**: Capstone academic metadata, architecture specifications, strict zero data leakage safeguards, and compliance disclosures.

---

## 🔬 Reproducibility & AI Assistance Disclosure

- **Reproducibility**: All experiments use a fixed seed (`random_state=42`).
- **AI Assistance Disclosure**: Generative AI tools (Google Antigravity Assistant) were utilized for code scaffolding, documentation structuring, and bug fixing in full compliance with the 23CSE301 Capstone Guidelines. All data pipelines, mathematical formulas, and clinical insights were reviewed and validated by the student team.
