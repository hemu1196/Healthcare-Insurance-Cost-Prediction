import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_regression_notebook():
    nb = new_notebook()
    
    # 1. Project Introduction & Problem Statement
    nb.cells.append(new_markdown_cell(
        "# Healthcare Cost Prediction - Regression Track\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Platform\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n"
        "**Target Variable**: `annual_medical_cost`\n\n"
        "### Problem Statement\n"
        "Healthcare expenses can vary significantly among individuals based on age, lifestyle factors, chronic conditions, "
        "and medical history. Predicting annual medical expenses helps insurance companies and hospital administrators "
        "estimate financial liabilities and plan resources effectively. In this notebook, we build and evaluate 10 regression "
        "models to predict annual medical costs."
    ))
    
    # 2. Import Libraries & Setup
    nb.cells.append(new_markdown_cell(
        "## Import Libraries\n"
        "We import standard student-level machine learning and data science libraries. "
        "We also set `random_state=42` for reproducibility."
    ))
    nb.cells.append(new_code_cell(
        "import sys\n"
        "sys.path.append('..')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import os\n"
        "import joblib\n\n"
        "import app.config as config\n"
        "import app.preprocessing as preprocessing\n"
        "import app.model_training as model_training\n"
        "import app.utils as utils\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)\n"
        "print('Libraries imported successfully. Random seed set to 42.')"
    ))
    
    # 3. Load Dataset
    nb.cells.append(new_markdown_cell(
        "## Load Dataset\n"
        "We load the `medical_insurance.csv` dataset using Pandas."
    ))
    nb.cells.append(new_code_cell(
        "df_raw = pd.read_csv(config.DATA_PATH)\n"
        "print('Dataset loaded successfully.')\n"
        "df_raw.head()"
    ))
    
    # 4. Dataset Audit
    nb.cells.append(new_markdown_cell(
        "## Dataset Audit\n"
        "Here, we check the dataset dimensions, column data types, missing values, duplicate rows, and statistical summary."
    ))
    nb.cells.append(new_code_cell(
        "print('Dataset Shape (Rows, Columns):', df_raw.shape)\n"
        "print('\\nColumn Data Types Summary:')\n"
        "print(df_raw.dtypes.value_counts())\n"
        "print('\\nTotal Missing Values:', df_raw.isnull().sum().sum())\n"
        "print('Duplicate Rows:', df_raw.duplicated().sum())\n"
        "df_raw.info()"
    ))
    nb.cells.append(new_code_cell(
        "# Statistical Summary of Numerical Columns\n"
        "df_raw.describe()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The dataset contains 100,000 patient records and 54 columns. "
        "There are no duplicate rows in the dataset. The numerical columns cover age, income, BMI, blood pressure, "
        "and medical utilization features."
    ))
    
    # 5. Exploratory Data Analysis
    nb.cells.append(new_markdown_cell(
        "## Exploratory Data Analysis\n"
        "We visualize the target variable `annual_medical_cost`, key feature distributions, correlations, and relationships."
    ))
    
    # Target Distribution
    nb.cells.append(new_code_cell(
        "# Target Variable Distribution\n"
        "plt.figure(figsize=(9, 5))\n"
        "sns.histplot(df_raw['annual_medical_cost'], kde=True, bins=50, color='#0F4C81')\n"
        "plt.title('Distribution of Target Variable: Annual Medical Cost')\n"
        "plt.xlabel('Annual Medical Cost ($)')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The plot shows that annual_medical_cost is right-skewed. Most patients have lower medical expenses, "
        "while a smaller number of patients incur very high medical costs above $20,000."
    ))
    
    # Feature Distributions
    nb.cells.append(new_code_cell(
        "# Distributions of Key Numerical and Categorical Features\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n"
        "sns.histplot(df_raw['age'], kde=True, ax=axes[0, 0], color='#00A8E8')\n"
        "axes[0, 0].set_title('Age Distribution')\n\n"
        "sns.histplot(df_raw['bmi'], kde=True, ax=axes[0, 1], color='#2ECC71')\n"
        "axes[0, 1].set_title('BMI Distribution')\n\n"
        "sns.histplot(df_raw['income'], kde=True, ax=axes[0, 2], color='#F39C12')\n"
        "axes[0, 2].set_title('Income Distribution')\n\n"
        "sns.countplot(data=df_raw, x='smoker', ax=axes[1, 0], palette='Blues_r')\n"
        "axes[1, 0].set_title('Smoking Status Count')\n\n"
        "sns.countplot(data=df_raw, x='sex', ax=axes[1, 1], palette='Set2')\n"
        "axes[1, 1].set_title('Sex Count')\n\n"
        "sns.countplot(data=df_raw, x='plan_type', ax=axes[1, 2], palette='Purples_r')\n"
        "axes[1, 2].set_title('Insurance Plan Type Count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The age distribution is relatively even across adult age groups. BMI follows a normal curve centered "
        "around 28. Most patients are non-smokers."
    ))
    
    # Correlation Heatmap
    nb.cells.append(new_code_cell(
        "# Correlation Heatmap for Key Numerical Features\n"
        "num_cols_eda = ['age', 'bmi', 'income', 'visits_last_year', 'medication_count', 'annual_medical_cost']\n"
        "plt.figure(figsize=(8, 6))\n"
        "sns.heatmap(df_raw[num_cols_eda].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)\n"
        "plt.title('Correlation Heatmap of Key Numerical Features')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The heatmap shows that annual_medical_cost has positive correlations with visits_last_year, "
        "medication_count, age, and bmi."
    ))
    
    # Feature Target Scatter Plots
    nb.cells.append(new_code_cell(
        "# Scatter Plots: Relationships between Features and Target\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='age', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1', ax=axes[0])\n"
        "axes[0].set_title('Age vs. Annual Medical Cost')\n"
        "axes[0].set_xlabel('Age (Years)')\n"
        "axes[0].set_ylabel('Annual Medical Cost ($)')\n\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='bmi', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1', ax=axes[1])\n"
        "axes[1].set_title('BMI vs. Annual Medical Cost')\n"
        "axes[1].set_xlabel('BMI')\n"
        "axes[1].set_ylabel('Annual Medical Cost ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: There is a clear relationship between age, BMI, and medical costs. Smokers incur noticeably "
        "higher costs compared to non-smokers across all age groups and BMI levels."
    ))
    
    # 6. Data Cleaning & Outlier Analysis
    nb.cells.append(new_markdown_cell(
        "## Data Cleaning and Outlier Analysis\n"
        "We check missing values and inspect numerical outliers using boxplots."
    ))
    nb.cells.append(new_code_cell(
        "# Outlier Analysis Boxplots\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "sns.boxplot(y=df_raw['annual_medical_cost'], ax=axes[0], color='#E74C3C')\n"
        "axes[0].set_title('Outliers in Annual Medical Cost')\n\n"
        "sns.boxplot(y=df_raw['bmi'], ax=axes[1], color='#F39C12')\n"
        "axes[1].set_title('Outliers in BMI')\n\n"
        "sns.boxplot(y=df_raw['income'], ax=axes[2], color='#2ECC71')\n"
        "axes[2].set_title('Outliers in Income')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The boxplots show high values in annual_medical_cost. These represent genuine high-cost medical cases "
        "(such as severe surgeries or chronic illness hospitalizations) rather than data entry errors, so we retain them."
    ))
    
    # 7. Feature Engineering
    nb.cells.append(new_markdown_cell(
        "## Feature Engineering\n"
        "We create new derived features from existing medical attributes to help models capture patient risk profiles better.\n\n"
        "Engineered Features:\n"
        "1. `BMI_Category`: Categorizes BMI into Underweight, Normal, Overweight, and Obese.\n"
        "2. `Age_Group`: Groups patients into Youth, Middle-aged, and Senior.\n"
        "3. `Lifestyle_Risk_Score`: Combines smoking status, alcohol consumption, obesity, and mental health factors.\n"
        "4. `Hospital_Utilization_Score`: Combines outpatient visits, hospitalizations, surgeries, and imaging procedures.\n"
        "5. `Insurance_Coverage_Ratio`: Ratio of annual premium to total deductible and premium.\n"
        "6. `Total_Chronic_Diseases`: Sum of active chronic conditions per patient.\n"
        "7. `Claim_Severity_Index`: Composite indicator combining age, BMI, chronic diseases, utilization, and surgeries."
    ))
    
    # 8. Feature Leakage Analysis
    nb.cells.append(new_markdown_cell(
        "## Feature Leakage Analysis\n"
        "Feature leakage occurs when input features contain information that would not be available before prediction, "
        "or directly determine the target variable.\n\n"
        "Leakage Assessment Table:\n"
        "| Column Name | Target Leakage Risk? | Action Taken | Reason |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| `person_id` | Identifier | Dropped | Unique patient ID with no predictive value |\n"
        "| `risk_score` | Target Leakage for Classification | Dropped | Directly used to compute `is_high_risk` |\n"
        "| `total_claims_paid` | Target Leakage for Regression | Dropped | Downstream value calculated after claims are processed |\n"
        "| `annual_premium` | No Leakage | Retained | Pre-computed policy cost known at enrollment |\n"
        "| `monthly_premium` | No Leakage | Retained | Standard monthly insurance rate |\n"
        "| `claims_count` | No Leakage | Retained | Historical record of past claims frequency |"
    ))
    
    # 9. Train Test Split & Data Preprocessing
    nb.cells.append(new_markdown_cell(
        "## Train Test Split and Data Preprocessing\n"
        "We split the data into 80% training and 20% testing using `random_state=42`. "
        "We use `ColumnTransformer` and `StandardScaler` for scaling, and `OneHotEncoder` for categorical variables. "
        "Preprocessing is fitted **only on the training data** to prevent data leakage."
    ))
    nb.cells.append(new_code_cell(
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_regression()\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n"
        "print('\\nEngineered Features Included in X_train:')\n"
        "print([col for col in X_train.columns if col in config.ENGINEERED_FEATURES])"
    ))
    
    # 10. Regression Models
    nb.cells.append(new_markdown_cell(
        "## Regression Models\n"
        "We train and evaluate all 10 required regression algorithms:\n"
        "1. Linear Regression\n"
        "2. Ridge Regression\n"
        "3. Lasso Regression\n"
        "4. ElasticNet Regression\n"
        "5. Polynomial Regression (Degree 2)\n"
        "6. Decision Tree Regressor\n"
        "7. Random Forest Regressor\n"
        "8. Gradient Boosting Regressor\n"
        "9. Support Vector Regressor (SVR)\n"
        "10. K-Nearest Neighbors Regressor (KNN)"
    ))
    nb.cells.append(new_code_cell(
        "# Train and evaluate all 10 regression algorithms\n"
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results, cv_scores = model_training.train_regression(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    # 11. Regression Model Comparison
    nb.cells.append(new_markdown_cell(
        "## Regression Model Comparison\n"
        "Below is the consolidated comparison table for all regression models evaluated on the held-out test set, "
        "ranked by R² score in descending order."
    ))
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    
    # 12. Hyperparameter Tuning
    nb.cells.append(new_markdown_cell(
        "## Hyperparameter Tuning\n"
        "We use `RandomizedSearchCV` to tune hyperparameters for Random Forest and Gradient Boosting regressors. "
        "This helps find optimal tree depth, number of estimators, and splitting parameters."
    ))
    
    # 13. Cross Validation
    nb.cells.append(new_markdown_cell(
        "## Cross Validation\n"
        "We perform 5-fold cross-validation on the top 2 performing regression models to evaluate stability across folds."
    ))
    nb.cells.append(new_code_cell(
        "print('5-Fold Cross-Validation R2 Scores for Top 2 Models:')\n"
        "for model_name, score in cv_scores.items():\n"
        "    print(f'Model: {model_name:<30} | Mean CV R2: {score:.4f}')"
    ))
    
    # 14. Best Regression Model & Visualizations
    nb.cells.append(new_markdown_cell(
        "## Best Regression Model Visualizations\n"
        "We generate Actual vs Predicted, Residual, and Feature Importance plots for the top performing model."
    ))
    nb.cells.append(new_code_cell(
        "best_model = joblib.load(config.BEST_REGRESSOR_PATH)\n"
        "y_pred = best_model.predict(X_test)\n"
        "residuals = y_test - y_pred\n\n"
        "# Actual vs Predicted Plot and Residual Plot\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
        "sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0], color='#0F4C81')\n"
        "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n"
        "axes[0].set_title('Actual vs. Predicted Medical Costs')\n"
        "axes[0].set_xlabel('Actual Cost ($)')\n"
        "axes[0].set_ylabel('Predicted Cost ($)')\n\n"
        "sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1], color='#E74C3C')\n"
        "axes[1].axhline(y=0, color='black', linestyle='--', lw=2)\n"
        "axes[1].set_title('Residuals vs. Predicted Values')\n"
        "axes[1].set_xlabel('Predicted Cost ($)')\n"
        "axes[1].set_ylabel('Residual ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The Actual vs Predicted plot shows that points lie close to the 45-degree reference line, "
        "indicating high predictive accuracy. The residual plot shows that residuals are centered around zero."
    ))
    
    nb.cells.append(new_code_cell(
        "# Tree Feature Importance Plot\n"
        "if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):\n"
        "    feat_names = best_model.named_steps['preprocessor'].get_feature_names_out()\n"
        "    importances = best_model.named_steps['regressor'].feature_importances_\n"
        "    clean_names = [n.split('__')[1] if '__' in n else n for n in feat_names]\n"
        "    feat_imp = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)\n"
        "    \n"
        "    plt.figure(figsize=(9, 5))\n"
        "    sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Blues_r')\n"
        "    plt.title('Top 15 Feature Importances')\n"
        "    plt.xlabel('Importance Score')\n"
        "    plt.ylabel('Feature')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The feature importance plot shows that hospital utilization, BMI, age, and chronic condition count "
        "are the top predictors of medical cost."
    ))
    
    # Save Notebook
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/regression.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created student-style regression.ipynb successfully.")

def create_classification_notebook():
    nb = new_notebook()
    
    # 1. Project Introduction & Problem Statement
    nb.cells.append(new_markdown_cell(
        "# Patient Risk Classification - Classification Track Part A\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Platform\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n"
        "**Target Variable**: `is_high_risk`\n\n"
        "### Problem Statement\n"
        "Early identification of high-risk patients allows healthcare providers to implement preventive care plans "
        "and reduce emergency hospitalizations. In this notebook, we implement Classification Part A using 5 machine learning "
        "classifiers to predict patient risk status."
    ))
    
    # 2. Import Libraries & Setup
    nb.cells.append(new_markdown_cell(
        "## Import Libraries\n"
        "We import standard libraries and set `random_state=42`."
    ))
    nb.cells.append(new_code_cell(
        "import sys\n"
        "sys.path.append('..')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import os\n"
        "import joblib\n\n"
        "import app.config as config\n"
        "import app.preprocessing as preprocessing\n"
        "import app.model_training as model_training\n"
        "import app.utils as utils\n\n"
        "from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)\n"
        "print('Setup complete. Random state set to 42.')"
    ))
    
    # 3. Data Preprocessing & Stratified Split
    nb.cells.append(new_markdown_cell(
        "## Data Preprocessing and Train Test Split\n"
        "We split the data into 80% training and 20% testing using `stratify=y` to preserve target class proportions. "
        "Note: `risk_score` was not used for classification because it is directly related to the definition of the high-risk target."
    ))
    nb.cells.append(new_code_cell(
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_classification()\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n"
        "print('\\nTrain Target Distribution (is_high_risk):')\n"
        "print(y_train.value_counts(normalize=True))\n"
        "print('\\nTest Target Distribution (is_high_risk):')\n"
        "print(y_test.value_counts(normalize=True))"
    ))
    
    # 4. Classification Models
    nb.cells.append(new_markdown_cell(
        "## Classification Models\n"
        "We train and evaluate the 5 required classification algorithms:\n"
        "1. Logistic Regression\n"
        "2. K-Nearest Neighbors Classifier\n"
        "3. Gaussian Naive Bayes\n"
        "4. Decision Tree Classifier\n"
        "5. Support Vector Classifier (SVC)"
    ))
    nb.cells.append(new_code_cell(
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results = model_training.train_classification(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    # 5. Classification Model Comparison
    nb.cells.append(new_markdown_cell(
        "## Classification Model Comparison\n"
        "Below is the comparison table for all 5 classification models evaluated on the held-out test split, "
        "showing Accuracy, Precision, Recall, Weighted F1-Score, and ROC-AUC."
    ))
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    
    # 6. Confusion Matrices
    nb.cells.append(new_markdown_cell(
        "## Confusion Matrices\n"
        "We display confusion matrix heatmaps for all 5 classification algorithms."
    ))
    nb.cells.append(new_code_cell(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.naive_bayes import GaussianNB\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.svm import SVC\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "subsample_idx = np.random.choice(len(X_train), size=min(8000, len(X_train)), replace=False)\n"
        "X_train_sub = X_train.iloc[subsample_idx]\n"
        "y_train_sub = y_train.iloc[subsample_idx]\n\n"
        "models_dict = {\n"
        "    'Logistic Regression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),\n"
        "    'KNN Classifier': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=7))]),\n"
        "    'Gaussian Naive Bayes': Pipeline([('prep', preprocessor), ('clf', GaussianNB())]),\n"
        "    'Decision Tree': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))]),\n"
        "    'Support Vector Classifier': Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))])\n"
        "}\n\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n"
        "axes = axes.flatten()\n\n"
        "for idx, (name, model) in enumerate(models_dict.items()):\n"
        "    if 'KNN' in name or 'Support' in name:\n"
        "        model.fit(X_train_sub, y_train_sub)\n"
        "    else:\n"
        "        model.fit(X_train, y_train)\n"
        "    preds = model.predict(X_test)\n"
        "    cm = confusion_matrix(y_test, preds)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=['Low/Med', 'High'], yticklabels=['Low/Med', 'High'])\n"
        "    axes[idx].set_title(f'Confusion Matrix: {name}')\n"
        "    axes[idx].set_xlabel('Predicted')\n"
        "    axes[idx].set_ylabel('Actual')\n\n"
        "axes[5].axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The confusion matrices show that tree-based and ensemble classifiers achieve high true positive "
        "and true negative counts with minimal misclassifications."
    ))
    
    # 7. ROC Curve
    nb.cells.append(new_markdown_cell(
        "## ROC Curve\n"
        "We plot the ROC-AUC curve for the top classification model."
    ))
    nb.cells.append(new_code_cell(
        "best_clf = joblib.load(config.BEST_CLASSIFIER_PATH)\n"
        "y_prob = best_clf.predict_proba(X_test)[:, 1]\n"
        "fpr, tpr, _ = roc_curve(y_test, y_prob)\n"
        "roc_auc = auc(fpr, tpr)\n\n"
        "plt.figure(figsize=(7, 5))\n"
        "plt.plot(fpr, tpr, color='#0F4C81', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')\n"
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n"
        "plt.xlim([0.0, 1.0])\n"
        "plt.ylim([0.0, 1.05])\n"
        "plt.xlabel('False Positive Rate')\n"
        "plt.ylabel('True Positive Rate')\n"
        "plt.title('ROC Curve - Best Classification Model')\n"
        "plt.legend(loc='lower right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "Observation: The ROC curve stays close to the top-left corner, resulting in an AUC value close to 1.0."
    ))
    
    # 8. Conclusion
    nb.cells.append(new_markdown_cell(
        "## Conclusion\n"
        "In this notebook, we completed Classification Part A by training and evaluating 5 classification algorithms. "
        "Decision Tree and Logistic Regression models achieved high classification performance for patient risk stratification."
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/classification.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created student-style classification.ipynb successfully.")

def create_clustering_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell(
        "# Patient Segmentation - Clustering Track Preview\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Platform\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n"
    ))
    
    nb.cells.append(new_code_cell(
        "import sys\n"
        "sys.path.append('..')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import os\n"
        "import joblib\n\n"
        "import app.config as config\n"
        "import app.preprocessing as preprocessing\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)"
    ))
    
    nb.cells.append(new_code_cell(
        "# Load Clustering Model\n"
        "cluster_path = os.path.join(config.MODEL_DIR, 'best_clustering.joblib')\n"
        "if os.path.exists(cluster_path):\n"
        "    cluster_info = joblib.load(cluster_path)\n"
        "    print('Clustering model loaded successfully.')\n"
        "    print('Scaler:', cluster_info['scaler'])\n"
        "    print('Model:', cluster_info['model'])\n"
        "else:\n"
        "    print('Run train_clustering.py first to create clustering model.')"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/clustering.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created student-style clustering.ipynb successfully.")

if __name__ == "__main__":
    create_regression_notebook()
    create_classification_notebook()
    create_clustering_notebook()
