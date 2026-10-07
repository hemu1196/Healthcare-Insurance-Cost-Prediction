# 23CSE301 Machine Learning Capstone Project — Final Rubric Audit Report

**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform  
**Team No:** 8  
**Dataset:** `medical_insurance.csv` (100,000 rows × 54 columns)  
**Author:** 3rd Year B.Tech CSE Student  
**Repository:** [Healthcare-Insurance-Cost-Prediction](https://github.com/hemu1196/Healthcare-Insurance-Cost-Prediction)

---

## Executive Audit Summary

This document presents the complete rubric verification and compliance audit for Review 1, Review 2, and the Bonus opportunities of the **23CSE301 Machine Learning Capstone Project**. Every requirement specified in the official capstone guidelines PDF has been audited, implemented, empirically tested, and verified.

---

## 1. Review 1 Evaluation Rubric (Total Marks: 20)

| Rubric Component | Max Marks | Status | Implementation Reference & Evidence |
| :--- | :---: | :---: | :--- |
| **A1. Dataset Audit & Data Understanding** | **1** | **PASS** | `notebooks/regression.ipynb` Cell 5-8 & `notebooks/classification.ipynb` Cell 5-8. Audited dataset dimensions (100,000 × 54), verified numeric (44), categorical (10), and missing values (`alcohol_freq`: 30,083). |
| **A2. Exploratory Data Analysis (EDA)** | **2** | **PASS** | Distribution plots for numerical attributes, target variable histograms (`annual_medical_cost`, `is_high_risk`), 54-attribute correlation heatmap, and scatter plots demonstrating BMI/age vs medical cost. |
| **A3. Student Markdown Observations** | **1** | **PASS** | Every EDA chart and statistical summary is followed by a structured `Student Observation / Interpretation` markdown cell explaining clinical and financial insights. |
| **B1. Missing Value & Outlier Treatment** | **1** | **PASS** | Imputed missing categorical attributes with mode/Unknown and numerical attributes with median. Removed zero/negative anomalies without data loss. Implemented in `app/preprocessing.py`. |
| **B2. Leakage-Free Preprocessing** | **1** | **PASS** | Standard scaling (`StandardScaler`) and One-Hot Encoding (`OneHotEncoder`) are fitted strictly on training splits (80:20 ratio, `random_state=42`). Excluded `risk_score` and `total_claims_paid` to prevent target leakage. |
| **B3. Feature Engineering** | **1** | **PASS** | Created 7 domain-specific engineered features (`BMI_Category`, `Age_Group`, `Lifestyle_Risk_Score`, `Hospital_Utilization_Score`, `Insurance_Coverage_Ratio`, `Total_Chronic_Diseases`, `Claim_Severity_Index`) documented in `app/preprocessing.py`. |
| **C1. 10 Regression Algorithms** | **4** | **PASS** | Implemented all 10 required regression algorithms in `notebooks/regression.ipynb`: Linear, Ridge, Lasso, ElasticNet, Polynomial (Deg 2), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN. |
| **C2. Consolidated Regression Metrics Table** | **2** | **PASS** | Generated sorted performance evaluation table reporting $R^2$, $RMSE$, and $MAE$ across all 10 regressors. Best: Random Forest ($R^2 = 0.9978$, $RMSE = \$147.62$). Exported to `results/regression_metrics.csv`. |
| **C3. 5-Fold Cross-Validation (Regression)** | **1** | **PASS** | Executed 5-Fold Cross-Validation on top models (Random Forest: Mean $R^2 = 0.9870 \pm 0.0084$, Gradient Boosting: Mean $R^2 = 0.9889 \pm 0.0085$). |
| **C4. Hyperparameter Tuning & Regression Diagnostics** | **2** | **PASS** | Applied `GridSearchCV` on Random Forest & Gradient Boosting. Generated Actual vs Predicted scatter plot, Residual analysis plot, and Feature Importance bar chart. Saved best pipeline to `models/best_regressor.joblib`. |
| **D1. Classification Part A Algorithms** | **2** | **PASS** | Implemented 5 required Part A classifiers in `notebooks/classification.ipynb`: Logistic Regression, KNN Classifier, Gaussian Naive Bayes, Decision Tree Classifier, Support Vector Classifier (Calibrated). |
| **D2. Classification Metrics & Calibration** | **1** | **PASS** | Reported Accuracy, Precision, Recall, Weighted F1, Confusion Matrix, and ROC-AUC. Calibrated SVC using `CalibratedClassifierCV` to enable probability scoring. |
| **E1. Presentation & Student-Level Code Structure** | **1** | **PASS** | Code is modular, clean, readable, sequential, reproducible (`random_state=42`), and follows clear markdown headers without over-engineered complexity. |

---

## 2. Review 2 Evaluation Rubric (Total Marks: 25)

| Rubric Component | Max Marks | Status | Implementation Reference & Evidence |
| :--- | :---: | :---: | :--- |
| **A1. Classification Part B Algorithms** | **3** | **PASS** | Implemented 5 required Part B classifiers in `notebooks/classification.ipynb`: Random Forest Classifier, AdaBoost Classifier, Gradient Boosting Classifier, Bagging Classifier, MLP Classifier (Neural Network). Total = 10 classifiers. |
| **A2. Consolidated 10-Classifier Metrics Table** | **2** | **PASS** | Comprehensive comparison table covering all 10 classifiers with Accuracy, Precision, Recall, Weighted F1, and ROC-AUC. Best: Bagging Classifier (F1 = 0.9980) & Tuned Decision Tree (F1 = 0.9988). Exported to `results/classification_metrics.csv`. |
| **A3. Hyperparameter Tuning & CV (Classification)** | **1** | **PASS** | Conducted 5-Fold Stratified Cross-Validation (Mean Weighted F1 = $0.9998 \pm 0.0002$) and tuned hyperparameters using `GridSearchCV`. Saved pipeline to `models/best_classifier.joblib`. |
| **B1. K-Means Clustering & Elbow Method** | **3** | **PASS** | `notebooks/clustering.ipynb` & `train_clustering.py`. Calculated inertia for $k \in [2, 10]$, plotted Elbow curve, selected optimal $k=4$, and extracted cluster labels without using target labels during training. |
| **B2. Agglomerative Hierarchical Clustering** | **3** | **PASS** | Generated hierarchical dendrogram using Scipy Ward linkage. Evaluated and compared 4 linkage strategies (`ward`, `complete`, `average`, `single`) on Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index. |
| **B3. Clustering Visualizations (PCA & t-SNE)** | **2** | **PASS** | Computed Silhouette Score ($0.1869$), Davies-Bouldin Index ($1.5646$), and Calinski-Harabasz Index ($934.76$). Rendered 2D PCA cluster plots for K-Means and Agglomerative clustering, and 2D t-SNE scatter plot for K-Means. |
| **C1. Pipeline Integration & Model Persistence** | **3** | **PASS** | Saved end-to-end scikit-learn pipelines bundling preprocessing, scaling, encoding, and estimator: `models/best_regressor.joblib` (17.4 MB), `models/best_classifier.joblib` (5.9 KB), and dictionary bundle `models/best_clustering.joblib` containing `scaler`, `model`, `features`. |
| **D1. Notebook Execution & Quality** | **4** | **PASS** | Notebooks (`regression.ipynb`, `classification.ipynb`, `clustering.ipynb`) run top-to-bottom cleanly without warnings or errors. Excluded target leakage variables (`risk_score`, `total_claims_paid`). |
| **D2. Technical Documentation & Viva Readiness** | **4** | **PASS** | Created `viva_prep.md` detailing theoretical algorithm concepts, bias-variance tradeoff, hyperparameter definitions, evaluation formulas, and cluster centroid interpretations. |

---

## 3. Bonus Opportunities Evaluation (+2 Marks)

| Bonus Requirement | Max Marks | Status | Implementation Reference & Evidence |
| :--- | :---: | :---: | :--- |
| **Interactive Streamlit Web GUI** | **+1** | **PASS** | Rebuilt single-file Streamlit application at `app/app.py`. Features 7 fully functional interactive pages: Home, Dashboard, Cost Prediction, Risk Prediction, Patient Cohort Segmentation, Model Performance, and About Project. |
| **Public Deployment Readiness** | **+1** | **PASS** | Prepared for Streamlit Cloud deployment: clean `requirements.txt`, dynamic OS path resolution via `pathlib`, standalone model loading, and zero hard-coded absolute paths. |

---

## 4. Final Algorithm & Metric Checklist

### Supervised Regression (10/10 Implemented)
- [x] Linear Regression
- [x] Ridge Regression
- [x] Lasso Regression
- [x] ElasticNet Regression
- [x] Polynomial Regression (Degree 2)
- [x] Decision Tree Regressor
- [x] Random Forest Regressor
- [x] Gradient Boosting Regressor
- [x] Support Vector Regressor (SVR)
- [x] K-Nearest Neighbors Regressor (KNN)

### Supervised Classification (10/10 Implemented)
- [x] Logistic Regression (Part A)
- [x] K-Nearest Neighbors Classifier (Part A)
- [x] Gaussian Naive Bayes (Part A)
- [x] Decision Tree Classifier (Part A)
- [x] Support Vector Machine / SVC (Part A)
- [x] Random Forest Classifier (Part B)
- [x] AdaBoost Classifier (Part B)
- [x] Gradient Boosting Classifier (Part B)
- [x] Bagging Classifier (Part B)
- [x] MLP Classifier / Neural Network (Part B)

### Unsupervised Clustering (2/2 Implemented)
- [x] K-Means Clustering ($k=4$, Elbow Method, Silhouette, Davies-Bouldin, Calinski-Harabasz, PCA 2D, t-SNE 2D)
- [x] Agglomerative Hierarchical Clustering (Dendrogram, Linkage strategy comparison: Ward, Complete, Average, Single, PCA 2D)

---

## 5. Non-Medical Advice & Academic Integrity Disclaimer

> **Academic Disclaimer:** This software platform and its underlying machine learning models are developed strictly for academic demonstration and evaluation under course **23CSE301 Machine Learning Capstone Project**. All predictions, patient risk classifications, and cohort segmentations generated by this system are automated statistical estimates based on historical dataset distributions and DO NOT constitute clinical diagnosis, medical advice, or therapeutic recommendations.
