# Healthcare Cost Prediction and Patient Risk Intelligence Platform
 
**Course:** 23CSE301 Machine Learning – Capstone Project  
**Team No:** 8  
**Section:** CSE-E  
**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform  
**Dataset:** Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)  
**Dataset Dimensions:** 100,000 rows × 54 columns  
**University:** Amrita Vishwa Vidyapeetham  
**Academic Year:** 2026–2027  

**Live Streamlit Application:** https://healthcare-insurance-cost-prediction-mekvzuvgfimmhccezb5qjp.streamlit.app/

**GitHub Repository:** https://github.com/hemu1196/Healthcare-Insurance-Cost-Prediction

---

## 1. Project Overview

This project focuses on using Machine Learning to analyse healthcare insurance data and predict patient-related outcomes.

We implemented three main Machine Learning tracks:

1. **Regression:** Predict the annual medical cost of a patient.
2. **Classification:** Predict whether a patient belongs to the Low/Medium Risk or High Risk category.
3. **Clustering:** Group patients with similar characteristics without using predefined target labels.

We implemented **10 Regression algorithms, 10 Classification algorithms, and 3 Clustering methods**, making a total of **23 machine learning implementations**.

Along with the Jupyter Notebooks, we developed a **Streamlit web application** called MedPredict Analytics to demonstrate predictions, visualisations, and model comparisons.

The project follows the requirements of the **23CSE301 Machine Learning Capstone Project – Review 1 and Review 2**.

---

## 2. Team Members

| Student Name | Register Number | 
|---|---|---|
| P. Hema Chandra | CB.SC.U4CSE24437 |  
| K. Navadeep | CB.SC.U4CSE24466 |  
| Rohan C. | CB.SC.U4CSE24441 |  

All three team members are involved in understanding, testing, and presenting the complete project.

---

## 3. Problem Statement

Healthcare insurance datasets contain information about patient demographics, lifestyle habits, medical conditions, hospital visits, and insurance policies.

Analysing this information manually can be difficult because of the large number of records and attributes.

Our objective is to apply Machine Learning techniques to:

- Estimate annual patient medical expenses.
- Predict patient risk categories.
- Identify groups of patients with similar characteristics.
- Compare different Machine Learning algorithms.
- Display results and predictions through an interactive web application.

---

## 4. Project Objectives

The main objectives of this project are:

1. To load and understand the healthcare insurance dataset.
2. To perform Exploratory Data Analysis (EDA).
3. To identify and handle missing values and duplicate records.
4. To perform feature encoding, scaling, and feature engineering.
5. To implement all 10 required regression algorithms.
6. To implement all 10 required classification algorithms.
7. To implement K-Means and Agglomerative Hierarchical Clustering.
8. To implement Divisive Clustering as an additional comparison.
9. To evaluate algorithms using suitable performance metrics.
10. To perform cross-validation and hyperparameter tuning.
11. To generate model-specific visualisations and comparison graphs.
12. To build and deploy a Streamlit GUI for project demonstration.

---

## 5. Dataset Description

**Dataset Name:** Medical Insurance Cost Prediction Dataset  
**Dataset File:** `medical_insurance.csv`

| Dataset Information | Value |
|---|---|
| Total Records | 100,000 |
| Total Attributes | 54 |
| Total Missing Values | 30,083 |
| Duplicate Records | 0 |
| Regression Target | `annual_medical_cost` |
| Classification Target | `is_high_risk` |

### 5.1 Numerical Features

Examples include:

- `age`
- `bmi`
- `income`
- `annual_premium`
- `visits_last_year`
- `hospitalizations_last_3yrs`
- `medication_count`
- `systolic_bp`
- `diastolic_bp`

### 5.2 Categorical Features

Examples include:

- `sex`
- `region`
- `urban_rural`
- `education`
- `employment_status`
- `smoker`
- `alcohol_freq`
- `plan_type`

### 5.3 Binary Features

Examples include:

- `hypertension`
- `diabetes`
- `asthma`
- `kidney_disease`
- `liver_disease`
- `cancer_history`

### 5.4 Target Variables

**Regression Target: `annual_medical_cost`**

Represents the annual medical expenditure of a patient.

**Classification Target: `is_high_risk`**

- `0` – Low/Medium Risk
- `1` – High Risk

### 5.5 Dataset Sampling

The original dataset contains **100,000 records**.

For faster training and classroom demonstration, we used the following samples:

| Track | Sample Records |
|---|---:|
| Regression | 5,000 |
| Classification | 5,000 |
| Clustering | 1,200 |

For regression and classification, an 80:20 training and testing split is used within the selected samples.

The results reported below are from these experiments and do not represent training on the complete 100,000-record dataset.

---

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis helps us understand the dataset before training the Machine Learning models.

### 6.1 Dataset Inspection

We performed:

- Displaying the first and last records.
- Checking the dataset shape.
- Checking column names and data types.
- Identifying missing values.
- Identifying duplicate records.
- Viewing descriptive statistics.

### 6.2 Missing Value Analysis

We observed **30,083 missing values** in the `alcohol_freq` column.

Missing categorical values are handled using suitable preprocessing techniques.

We do not assume that missing alcohol-frequency values mean a patient does not consume alcohol.

### 6.3 EDA Visualisations

The notebooks include the following visualisations:

| Visualisation | Purpose |
|---|---|
| Histogram | Understand numerical feature distributions |
| Count Plot | Compare different categories |
| Boxplot | Identify possible outliers |
| Correlation Heatmap | Understand relationships between numerical variables |
| Scatter Plot | Observe relationships between features |
| Target Distribution | Understand the prediction target |

EDA helps us understand the characteristics and limitations of the dataset.

---

## 7. Data Preprocessing and Feature Engineering

Before applying Machine Learning algorithms, we preprocess the dataset.

### 7.1 Handling Missing Values

We use:

- Median imputation for numerical attributes.
- Most frequent category imputation for categorical attributes.

### 7.2 One-Hot Encoding

Categorical features are converted into numerical form using One-Hot Encoding.

### 7.3 Feature Scaling

StandardScaler is used to scale numerical features.

Scaling is particularly important for distance-based and margin-based algorithms such as KNN and SVM.

### 7.4 Feature Engineering

We created seven additional features:

| Feature | Description |
|---|---|
| `BMI_Category` | Categorises patients based on BMI |
| `Age_Group` | Divides patients into age groups |
| `Lifestyle_Risk_Score` | Combines selected lifestyle-related factors |
| `Hospital_Utilization_Score` | Combines hospital visits and procedures |
| `Insurance_Coverage_Ratio` | Calculates a ratio based on premium and deductible |
| `Total_Chronic_Diseases` | Counts selected chronic disease indicators |
| `Claim_Severity_Index` | Combines selected patient and hospital-related factors |

These features are created to represent potentially useful patterns in the dataset.

### 7.5 Data Leakage Prevention

We removed target columns, patient identifiers, and selected outcome-related attributes from the predictors.

For supervised learning, preprocessing is fitted using training data and then applied to testing data.

This reduces obvious data leakage, although possible indirect relationships between features and targets still require careful evaluation.

---

## 8. Regression Track

**Notebook:** `notebooks/regression.ipynb`

### 8.1 Objective

To predict the estimated annual medical cost using supervised Machine Learning algorithms.

### 8.2 Regression Algorithms Implemented

| No. | Algorithm |
|---|---|
| 1 | Linear Regression |
| 2 | Ridge Regression |
| 3 | Lasso Regression |
| 4 | ElasticNet Regression |
| 5 | Polynomial Regression |
| 6 | Decision Tree Regressor |
| 7 | Random Forest Regressor |
| 8 | Gradient Boosting Regressor |
| 9 | Support Vector Regressor (SVR) |
| 10 | K-Nearest Neighbors Regressor |

### 8.3 Regression Evaluation Metrics

**R² Score**

Measures how much variation in the target is explained by the model. Higher values are generally better.

**RMSE – Root Mean Squared Error**

Measures prediction error while giving more importance to larger errors. Lower values are better.

**MAE – Mean Absolute Error**

Measures the average absolute difference between actual and predicted values. Lower values are better.

### 8.4 Regression Results

Based on the recorded results in our regression notebook:

| Selected Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| Lasso Regression | 0.1998 | 2667.94 | 1692.05 |

**Selected Model:** Lasso Regression

Lasso achieved the highest recorded held-out R² in this experiment.

An R² score of approximately 0.20 indicates that the model explains about 20% of the variation in annual medical costs in the test sample.

### 8.5 Regression Visualisations

Our notebook includes:

- Actual vs Predicted plots.
- Residual plots.
- R² comparison graphs.
- RMSE and MAE comparison graphs.
- Decision Tree visualisation.
- Feature importance graphs.
- Hyperparameter tuning results.
- Five-fold cross-validation results.

### 8.6 Hyperparameter Tuning

We applied GridSearchCV to selected regression models, including Ridge and Random Forest.

Five-fold cross-validation was also used to examine model performance across different training and validation splits.

---

## 9. Classification Track

**Notebook:** `notebooks/classification.ipynb`

### 9.1 Objective

To classify patients into Low/Medium Risk and High Risk categories.

### 9.2 Classification Algorithms Implemented

**Part A – Review 1**

| No. | Algorithm |
|---|---|
| 1 | Logistic Regression |
| 2 | K-Nearest Neighbors Classifier |
| 3 | Gaussian Naive Bayes |
| 4 | Decision Tree Classifier |
| 5 | Support Vector Machine (SVC) |

**Part B – Review 2**

| No. | Algorithm |
|---|---|
| 6 | Random Forest Classifier |
| 7 | AdaBoost Classifier |
| 8 | Gradient Boosting Classifier |
| 9 | Bagging Classifier |
| 10 | Multi-Layer Perceptron (MLP Classifier) |

### 9.3 Classification Evaluation Metrics

**Accuracy:** Proportion of correctly classified records.

**Precision:** Measures how many predicted cases in a class actually belong to that class.

**Recall:** Measures how many actual cases in a class are identified correctly.

**Weighted F1 Score:** Combines precision and recall while accounting for the number of samples in each class.

**ROC-AUC:** Measures how well the classifier separates the two risk categories across different thresholds.

### 9.4 Classification Results

Based on the recorded results in our classification notebook:

| Selected Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|
| Gradient Boosting Classifier | 0.9940 | 0.9940 | 0.9999 |

**Selected Model:** Gradient Boosting Classifier

Gradient Boosting achieved the highest recorded weighted F1 in this comparison.

These results show strong performance on the selected dataset sample. However, high classification scores should be checked for possible hidden data leakage and dataset-specific relationships.

### 9.5 Classification Visualisations

Our notebook includes:

- Confusion matrices.
- ROC curves.
- Accuracy comparison plots.
- Precision and Recall comparison plots.
- Weighted F1 comparison plots.
- Decision Tree visualisation.
- Feature importance graphs.
- MLP training-loss graph.
- Consolidated comparison of all 10 classifiers.

### 9.6 Hyperparameter Tuning

We performed cross-validation and hyperparameter tuning for selected classifiers.

In the recorded Random Forest tuning experiment, the mean cross-validated weighted F1 improved by approximately 0.004.

This helps demonstrate the importance of selecting suitable model parameters.

---

## 10. Clustering Track

**Notebook:** `notebooks/clustering.ipynb`

### 10.1 Objective

To group patients with similar characteristics without using predefined target labels during model fitting.

### 10.2 Clustering Algorithms Implemented

| No. | Algorithm | Approach |
|---|---|---|
| 1 | K-Means Clustering | Centroid-based |
| 2 | Agglomerative Hierarchical Clustering | Bottom-up |
| 3 | Divisive Clustering (Bisecting K-Means) | Top-down |

K-Means and Agglomerative Hierarchical Clustering are required by the course guidelines.

Divisive Clustering was implemented as an additional comparison.

### 10.3 K-Means Clustering

K-Means divides patients into clusters based on similarity.

We used the Elbow Method and Silhouette Score to examine different values of K.

### 10.4 Agglomerative Hierarchical Clustering

Agglomerative clustering begins by treating every patient as an individual cluster.

It gradually merges similar clusters.

We used a dendrogram to visualise the hierarchical merging process.

### 10.5 Divisive Clustering

Divisive clustering follows a top-down approach.

It starts with a larger group and divides it into smaller groups.

We implemented Bisecting K-Means as our divisive clustering method.

### 10.6 Clustering Evaluation Metrics

**Silhouette Score:** Measures how well points fit within their assigned clusters. Higher is better.

**Davies-Bouldin Index:** Measures cluster similarity relative to internal dispersion. Lower is better.

**Calinski-Harabasz Index:** Compares between-cluster separation with within-cluster dispersion. Higher is generally better.

### 10.7 Clustering Results

The clustering experiment used 1,200 sampled patients.

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| K-Means | 0.125 | 3.281 | 104.35 |
| Agglomerative | 0.052 | 3.690 | 76.45 |
| Divisive | 0.121 | 3.295 | 104.43 |

**Selected Clustering Method:** K-Means

K-Means achieved the highest Silhouette Score in this experiment.

However, the relatively low Silhouette Scores indicate that the clusters are not strongly separated.

### 10.8 Clustering Visualisations

Our notebook includes:

- Elbow Curve.
- Silhouette Score vs K.
- Agglomerative Dendrogram.
- PCA 2D Scatter Plots.
- t-SNE Visualisation.
- Cluster-size Bar Charts.
- Cluster Profile Heatmap.
- Comparison of Clustering Evaluation Metrics.

---

## 11. Streamlit Web Application

We developed a Streamlit web application called **MedPredict Analytics** to demonstrate our project.

### Live Streamlit Application

https://healthcare-insurance-cost-prediction-mekvzuvgfimmhccezb5qjp.streamlit.app/

### Main Features

**1. Home Dashboard**

Displays the dataset summary, total algorithms, target distributions, and model comparison graphs.

**2. Dataset & EDA**

Displays dataset information, missing-value analysis, descriptive statistics, and EDA visualisations.

**3. Healthcare Cost Prediction**

Allows users to enter patient information and obtain an estimated annual medical cost.

**4. Patient Risk Prediction**

Allows users to enter patient information and predict a risk category.

**5. Patient Segmentation**

Displays patient groups and clustering results.

**6. Model Performance**

Shows regression, classification, and clustering comparison results.

**7. About Project**

Displays project information, methods, and technologies.

### How the Streamlit Application Works

1. The user enters the required patient information.
2. Streamlit receives the input.
3. The saved preprocessing pipeline transforms the input.
4. The trained Machine Learning model makes a prediction.
5. The application displays the predicted result.

We used Joblib to save and load trained models so predictions can be generated without retraining the models every time.

**Note:** This application is developed for academic demonstration and should not be used for real clinical decisions.

---

## 12. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Creating graphs |
| Seaborn | Statistical visualisations |
| Scikit-learn | Machine Learning algorithms |
| SciPy | Hierarchical clustering |
| PCA | Dimensionality reduction |
| t-SNE | Cluster visualisation |
| Jupyter Notebook | Model training and evaluation |
| Streamlit | Interactive GUI |
| Joblib | Saving trained models |
| GitHub | Version control and project hosting |

---

## 13. Project Folder Structure

```text
Healthcare-Insurance-Cost-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── medical_insurance.csv
│
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
│
├── models/
│   └── Saved Machine Learning Models
│
├── results/
│   └── Model Comparison Tables
│
├── requirements.txt
└── README.md
```

---

## 14. How to Run the Project

### Step 1: Clone the GitHub Repository

```bash
git clone https://github.com/hemu1196/Healthcare-Insurance-Cost-Prediction.git
cd Healthcare-Insurance-Cost-Prediction
```

Alternatively, download the project as a ZIP file from GitHub and extract it.

### Step 2: Install Required Libraries

```bash
python3 -m pip install -r requirements.txt
```

### Step 3: Open the Jupyter Notebooks

```bash
jupyter notebook
```

Open and run:

```text
notebooks/regression.ipynb
notebooks/classification.ipynb
notebooks/clustering.ipynb
```

Run the notebook cells from top to bottom to generate the results and visualisations.

### Step 4: Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application normally opens at:

http://localhost:8501

You can also use our publicly deployed application:

https://healthcare-insurance-cost-prediction-mekvzuvgfimmhccezb5qjp.streamlit.app/

---

## 15. Review 2 Deliverables

Our project follows the official capstone evaluation guidelines.

| Deliverable | Implementation |
|---|---|
| D1 – Jupyter Notebooks | Three notebooks for the three tracks |
| D2 – GitHub Repository | Version control and meaningful commits |
| D3 – Comparison Tables | Regression, Classification, and Clustering |
| D4 – Visualisations | Model-specific graphs and EDA |
| D5 – README | Project description and instructions |
| D6 – Classification Comparison | All 10 classifiers |
| D7 – Clustering Visualisations | Elbow, Dendrogram, PCA, t-SNE |
| D8 – GUI / Deployment | Streamlit Web Application |

The working Streamlit GUI and public deployment are included for the optional bonus component.

---

## 16. Project Limitations

1. We used smaller samples to reduce training and execution time.
2. Regression performance is limited by the information available in the selected features.
3. Classification scores require additional validation for possible indirect target leakage.
4. Clustering results indicate relatively weak separation among groups.
5. Our application is intended for academic demonstration, not real medical diagnosis.

---

## 17. Future Improvements

We plan to:

1. Train and evaluate models on the complete dataset.
2. Improve feature selection and handle possible hidden data leakage.
3. Test models on independent datasets.
4. Explore additional techniques to improve cost prediction.
5. Improve cluster interpretation.
6. Add more features and improvements to the Streamlit interface.

---

## 18. Conclusion

In this Machine Learning Capstone Project, we implemented **10 Regression algorithms, 10 Classification algorithms, and 3 Clustering methods** for healthcare insurance data analysis.

We performed dataset inspection, EDA, data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and result visualisation.

Our classification models achieved strong results on the selected sample, while regression and clustering showed areas where further improvements are possible.

We also developed and deployed a Streamlit application to demonstrate the project interactively.

Through this project, we gained practical experience in supervised learning, unsupervised learning, model evaluation, data visualisation, and web application deployment.

---

### References

- 23CSE301 Machine Learning Capstone Project Guidelines
- Medical Insurance Cost Prediction Dataset
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib Documentation
- Seaborn Documentation
- Streamlit Documentation

---

**Developed by Team 08 – CSE-E**

**23CSE301 Machine Learning Capstone Project**  
**Amrita Vishwa Vidyapeetham**  
**Academic Year: 2026–2027**

**Project:** Healthcare Cost Prediction and Patient Risk Intelligence Platform

**Live Demo:** https://healthcare-insurance-cost-prediction-mekvzuvgfimmhccezb5qjp.streamlit.app/
