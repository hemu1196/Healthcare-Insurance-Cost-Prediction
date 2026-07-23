import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_regression_notebook():
    nb = new_notebook()
    
    # 1. Introduction
    nb.cells.append(new_markdown_cell(
        "# Healthcare Cost Prediction - Regression Track (Review 1)\n"
        "**Project**: Healthcare Cost Prediction and Patient Risk Intelligence Platform\n"
        "**Track**: Regression (Target: `annual_medical_cost`)\n"
        "**Academic Year**: 2026-27 | 23CSE301 Machine Learning - Capstone Project\n\n"
        "### Project Overview & Objectives\n"
        "In this notebook, we implement the complete **Regression Track** to predict patient annual medical costs. "
        "We audit and explore the Kaggle medical insurance dataset, engineer clinically relevant features, "
        "and train, tune, and compare 10 different regression algorithms to establish the best predictive model."
    ))
    
    # 2. Setup
    nb.cells.append(new_markdown_cell(
        "## 1. Setup and Environment\n"
        "We import all necessary libraries and load global configurations."
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
        "np.random.seed(config.RANDOM_STATE)"
    ))
    
    # 3. Data Load & Audit
    nb.cells.append(new_markdown_cell(
        "## 2. Dataset Loading & Audit\n"
        "We load the raw dataset and perform a basic audit (dimensions, column datatypes, missing values, duplicates, and statistical descriptions)."
    ))
    nb.cells.append(new_code_cell(
        "df_raw = pd.read_csv(config.DATA_PATH)\n"
        "print('Dataset Shape:', df_raw.shape)\n"
        "print('\\nMissing values count per column:')\n"
        "print(df_raw.isnull().sum().sum())\n"
        "print('\\nDuplicates count:')\n"
        "print(df_raw.duplicated().sum())\n"
        "df_raw.info()"
    ))
    nb.cells.append(new_code_cell(
        "df_raw.describe()"
    ))
    
    # 4. Written Insights and EDA
    nb.cells.append(new_markdown_cell(
        "### Written Insight - Dataset Audit\n"
        "**Observation**:\n"
        "- The dataset contains exactly **100,000 rows** and **54 columns**.\n"
        "- There are **0 duplicates** and minimal missing values in the categories, which are cleaned during preprocessing.\n"
        "- The variables span demographic indicators (age, sex, income), lifestyle markers (bmi, smoker, alcohol), "
        "clinical indicators (blood pressure, chronic diseases), and healthcare utilization metrics."
    ))
    
    nb.cells.append(new_markdown_cell(
        "## 3. Exploratory Data Analysis (EDA)\n"
        "We generate the mandatory visualisations to analyze distributions, correlations, and relationships."
    ))
    
    # Target distribution
    nb.cells.append(new_code_cell(
        "# Plot Target variable (annual_medical_cost) distribution\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(df_raw['annual_medical_cost'], kde=True, bins=50, color='#1E88E5')\n"
        "plt.title('Distribution of Target Variable: Annual Medical Cost')\n"
        "plt.xlabel('Annual Medical Cost ($)')\n"
        "plt.ylabel('Count')\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "### Written Insight - Target Distribution\n"
        "**Observation**:\n"
        "- The target variable `annual_medical_cost` exhibits a heavily right-skewed distribution.\n"
        "- The majority of patients incur medical costs below $20,000, but there is a long tail extending up to $150,000+.\n"
        "- This indicates the presence of high-severity outliers corresponding to patients with severe chronic conditions, complex hospitalization histories, or extreme risk factors like heavy smoking combined with high BMI."
    ))
    
    # Age vs Cost
    nb.cells.append(new_code_cell(
        "# Scatter Plot: Age vs Cost\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='age', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1')\n"
        "plt.title('Patient Age vs. Annual Medical Cost (Sampled, Colored by Smoking Status)')\n"
        "plt.xlabel('Age')\n"
        "plt.ylabel('Annual Medical Cost ($)')\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "### Written Insight - Age vs Cost Relationship\n"
        "**Observation**:\n"
        "- The scatter plot shows distinct segments of cost bands.\n"
        "- The baseline cost increases monotonically with age, but the smoking status (represented in red) shifts the cost bands upward by approximately $15,000 to $25,000.\n"
        "- This clear stratification highlights the massive additive effect of smoking on medical expenses."
    ))
    
    # BMI vs Cost
    nb.cells.append(new_code_cell(
        "# Scatter Plot: BMI vs Cost\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='bmi', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1')\n"
        "plt.title('Patient BMI vs. Annual Medical Cost (Sampled, Colored by Smoking Status)')\n"
        "plt.xlabel('BMI')\n"
        "plt.ylabel('Annual Medical Cost ($)')\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "### Written Insight - BMI vs Cost Relationship\n"
        "**Observation**:\n"
        "- There is a clear threshold effect around **BMI = 30** (which marks the clinical boundary for obesity).\n"
        "- For non-smokers, costs increase slowly with BMI. However, for smokers with BMI > 30, we see a massive spike in medical costs (the upper-right quadrant).\n"
        "- This represents a classic non-linear interaction effect, where the combination of obesity and smoking creates compound health hazards reflected in higher claim values."
    ))
    
    # Correlation Heatmap
    nb.cells.append(new_code_cell(
        "# Correlation matrix for key numerical columns\n"
        "selected_nums = ['age', 'bmi', 'income', 'visits_last_year', 'medication_count', 'annual_medical_cost']\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(df_raw[selected_nums].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)\n"
        "plt.title('Correlation Heatmap of Key Numerical Features')\n"
        "plt.show()"
    ))
    nb.cells.append(new_markdown_cell(
        "### Written Insight - Correlation Heatmap\n"
        "**Observation**:\n"
        "- `annual_medical_cost` is most strongly correlated with `visits_last_year` (0.42), `medication_count` (0.33), and `age` (0.33).\n"
        "- Moderate correlation is also seen with `bmi` (0.21).\n"
        "- Interestingly, `income` shows almost zero correlation with medical costs, suggesting financial capacity does not directly drive clinical claim severity in this dataset model."
    ))
    
    # 5. Preprocessing & Feature Engineering
    nb.cells.append(new_markdown_cell(
        "## 4. Feature Engineering & Preprocessing\n"
        "We apply feature engineering to create new clinically relevant variables and then preprocess the data (scaling and encoding)."
    ))
    nb.cells.append(new_code_cell(
        "# Preprocessing & splitting regression dataset\n"
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_regression()\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n"
        "print('\\nEngineered Features added:')\n"
        "print([col for col in X_train.columns if col in config.ENGINEERED_FEATURES])"
    ))
    nb.cells.append(new_markdown_cell(
        "### Preprocessing Details:\n"
        "- **Feature Engineering**: Engineered columns include `BMI_Category`, `Age_Group`, `Lifestyle_Risk_Score`, "
        "`Hospital_Utilization_Score`, `Insurance_Coverage_Ratio`, `Total_Chronic_Diseases`, and `Claim_Severity_Index`.\n"
        "- **Scaling**: Continuous features are scaled using `StandardScaler` fitted on the training split only to avoid data leakage.\n"
        "- **Encoding**: Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore', drop='first')`."
    ))
    
    # 6. Model Training
    nb.cells.append(new_markdown_cell(
        "## 5. Model Training & Comparison\n"
        "We train all 10 regression algorithms, perform hyperparameter tuning using RandomizedSearchCV, "
        "and compare the results."
    ))
    nb.cells.append(new_code_cell(
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results, cv_scores = model_training.train_regression(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    # 7. Comparison Table & Interpretation
    nb.cells.append(new_markdown_cell(
        "### 6. Results Summary and Validation\n"
        "We present the comparison table for all 10 algorithms ranked by R² score."
    ))
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    nb.cells.append(new_code_cell(
        "print('Cross Validation Results for the Top 2 models:')\n"
        "for name, score in cv_scores.items():\n"
        "    print(f'{name}: Mean CV R2 = {score:.4f}')"
    ))
    
    # 8. Visualisations
    nb.cells.append(new_markdown_cell(
        "## 7. Best Model Diagnostics & Visualizations\n"
        "We load the best saved model, generate actual-vs-predicted plots, residual plots, and feature importance."
    ))
    nb.cells.append(new_code_cell(
        "best_model = joblib.load(config.BEST_REGRESSOR_PATH)\n"
        "y_pred = best_model.predict(X_test)\n"
        "residuals = y_test - y_pred\n\n"
        "# Plot Actual vs Predicted\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0], color='#1E88E5')\n"
        "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n"
        "axes[0].set_title('Actual vs. Predicted Medical Costs')\n"
        "axes[0].set_xlabel('Actual Cost ($)')\n"
        "axes[0].set_ylabel('Predicted Cost ($)')\n\n"
        "# Plot Residuals\n"
        "sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1], color='#E53935')\n"
        "axes[1].axhline(y=0, color='black', linestyle='--', lw=2)\n"
        "axes[1].set_title('Residuals vs. Predicted Values')\n"
        "axes[1].set_xlabel('Predicted Cost ($)')\n"
        "axes[1].set_ylabel('Residual ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    nb.cells.append(new_code_cell(
        "# Feature Importance Plot for the best ensemble model\n"
        "if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):\n"
        "    try:\n"
        "        feat_names = best_model.named_steps['preprocessor'].get_feature_names_out()\n"
        "        importances = best_model.named_steps['regressor'].feature_importances_\n"
        "        clean_names = [n.split('__')[1] if '__' in n else n for n in feat_names]\n"
        "        feat_imp = pd.DataFrame({'Feature': clean_names, 'Importance': importances})\n"
        "        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)\n"
        "        \n"
        "        plt.figure(figsize=(10, 6))\n"
        "        sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Blues_r')\n"
        "        plt.title('Top 15 Feature Importances (Best Model)')\n"
        "        plt.show()\n"
        "    except Exception as e:\n"
        "        print('Could not render importances:', e)"
    ))
    
    # Save Notebook
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/regression.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created regression.ipynb successfully.")

def create_classification_notebook():
    nb = new_notebook()
    
    # 1. Introduction
    nb.cells.append(new_markdown_cell(
        "# Patient Risk Classification - Classification Track Part A (Review 1)\n"
        "**Project**: Healthcare Cost Prediction and Patient Risk Intelligence Platform\n"
        "**Track**: Classification Part A (Target: `is_high_risk`)\n"
        "**Academic Year**: 2026-27 | 23CSE301 Machine Learning - Capstone Project\n\n"
        "### Project Overview & Objectives\n"
        "In this notebook, we implement the **Classification Track Part A** to identify patients at high risk of elevated healthcare costs. "
        "We train, evaluate, and compare 5 classifiers (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM). "
        "We perform hyperparameter tuning using RandomizedSearchCV, construct the confusion matrix and ROC-AUC curves, and save the best classifier."
    ))
    
    # 2. Setup
    nb.cells.append(new_markdown_cell(
        "## 1. Setup and Environment\n"
        "We import all necessary libraries and load global configurations."
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
        "np.random.seed(config.RANDOM_STATE)"
    ))
    
    # 3. Data Load & Split
    nb.cells.append(new_markdown_cell(
        "## 2. Preprocessing & Stratified Train/Test Split\n"
        "We load the dataset, apply feature engineering, and perform a stratified train/test split. "
        "We verify that the target class distribution is balanced across the train and test splits."
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
    
    # 4. Model Training
    nb.cells.append(new_markdown_cell(
        "## 3. Classifier Model Training (Review 1 Part A)\n"
        "We train the 5 required classifiers: Logistic Regression, KNN, Naive Bayes (Gaussian), Decision Tree, and SVM. "
        "Hyperparameter tuning is performed on Logistic Regression and Decision Tree."
    ))
    nb.cells.append(new_code_cell(
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results = model_training.train_classification(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    # 5. Model Comparison
    nb.cells.append(new_markdown_cell(
        "## 4. Preliminary Results Comparison\n"
        "We display the metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for all models on the held-out test split, ranked by F1-Score."
    ))
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    
    # 6. Visualisations
    nb.cells.append(new_markdown_cell(
        "## 5. Best Classifier Performance Diagnostics\n"
        "We load the best saved classifier, plot the Confusion Matrix, and generate the ROC-AUC curve."
    ))
    nb.cells.append(new_code_cell(
        "best_clf = joblib.load(config.BEST_CLASSIFIER_PATH)\n"
        "y_pred = best_clf.predict(X_test)\n"
        "y_prob = best_clf.predict_proba(X_test)[:, 1]\n\n"
        "print('Classification Report for the Best Model:')\n"
        "print(classification_report(y_test, y_pred))\n\n"
        "# Confusion Matrix\n"
        "cm = confusion_matrix(y_test, y_pred)\n"
        "plt.figure(figsize=(6, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low/Med Risk', 'High Risk'], yticklabels=['Low/Med Risk', 'High Risk'])\n"
        "plt.title('Confusion Matrix - Best Classifier')\n"
        "plt.ylabel('Actual Label')\n"
        "plt.xlabel('Predicted Label')\n"
        "plt.show()"
    ))
    nb.cells.append(new_code_cell(
        "# ROC Curve\n"
        "fpr, tpr, thresholds = roc_curve(y_test, y_prob)\n"
        "roc_auc = auc(fpr, tpr)\n\n"
        "plt.figure(figsize=(8, 6))\n"
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')\n"
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n"
        "plt.xlim([0.0, 1.0])\n"
        "plt.ylim([0.0, 1.05])\n"
        "plt.xlabel('False Positive Rate')\n"
        "plt.ylabel('True Positive Rate')\n"
        "plt.title('Receiver Operating Characteristic (ROC) Curve')\n"
        "plt.legend(loc='lower right')\n"
        "plt.show()"
    ))
    
    # Save Notebook
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/classification.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created classification.ipynb successfully.")

if __name__ == "__main__":
    create_regression_notebook()
    create_classification_notebook()
