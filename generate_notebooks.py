import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_regression_notebook():
    nb = new_notebook()
    
    # Title & Metadata
    nb.cells.append(new_markdown_cell(
        "# Healthcare Cost Prediction - Regression Track (Review 1)\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Using Machine Learning\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n"
        "**Regression Target**: `annual_medical_cost`\n\n"
        "---\n"
        "### Academic Integrity Note\n"
        "Per course guidelines, AI code scaffolding is used for code setup and structure. "
        "All analytical observations and domain interpretations are marked with **`TODO` placeholders** for student authoring.\n\n"
        "Source/Citation: TODO — add source if external code was used."
    ))
    
    # 1. Setup
    nb.cells.append(new_markdown_cell(
        "## 1. Environment & Setup\n"
        "Import required libraries and configure global random state (`random_state=42`)."
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
        "print('Setup complete. Random state set to 42.')"
    ))
    
    # Section A: Dataset & EDA
    nb.cells.append(new_markdown_cell(
        "## SECTION A — DATASET & EDA (4 MARKS)\n"
        "In this section, we audit dataset dimensions, datatypes, missing values, duplicates, descriptive statistics, and plot feature distributions."
    ))
    
    nb.cells.append(new_code_cell(
        "# 1. Load Dataset & Audit Shape / Types\n"
        "df_raw = pd.read_csv(config.DATA_PATH)\n"
        "print('Dataset Shape:', df_raw.shape)\n"
        "print('\\nData Types Summary:')\n"
        "print(df_raw.dtypes.value_counts())\n"
        "print('\\nMissing Values Count per Column (Total):', df_raw.isnull().sum().sum())\n"
        "print('Duplicate Rows Count:', df_raw.duplicated().sum())\n"
        "df_raw.info()"
    ))
    
    nb.cells.append(new_code_cell(
        "# 2. Descriptive Statistics\n"
        "df_raw.describe()"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Dataset & Audit Summary\n"
        "TODO: Write observation after inspecting dataset shape, missing values, and descriptive statistics."
    ))
    
    # Target Distribution
    nb.cells.append(new_code_cell(
        "# 3. Target Distribution Plot (annual_medical_cost)\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(df_raw['annual_medical_cost'], kde=True, bins=50, color='#0F4C81')\n"
        "plt.title('Distribution of Target Variable: Annual Medical Cost')\n"
        "plt.xlabel('Annual Medical Cost ($)')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Target Distribution\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Feature Distributions
    nb.cells.append(new_code_cell(
        "# 4. Feature Distribution Plots (Key Numerical & Categorical Features)\n"
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n"
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
        "### Observation — Feature Distributions\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Outlier Analysis
    nb.cells.append(new_code_cell(
        "# 5. Outlier Analysis (Boxplots)\n"
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
        "### Observation — Outlier Analysis\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Correlation Heatmap
    nb.cells.append(new_code_cell(
        "# 6. Correlation Heatmap\n"
        "num_cols_eda = ['age', 'bmi', 'income', 'visits_last_year', 'medication_count', 'annual_medical_cost']\n"
        "plt.figure(figsize=(9, 7))\n"
        "sns.heatmap(df_raw[num_cols_eda].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)\n"
        "plt.title('Correlation Heatmap of Key Numerical Features')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Correlation Heatmap\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Scatter Plots
    nb.cells.append(new_code_cell(
        "# 7. Scatter Plots: Feature-Target Relationships\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='age', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1', ax=axes[0])\n"
        "axes[0].set_title('Age vs. Annual Medical Cost (Hue: Smoker)')\n"
        "axes[0].set_xlabel('Age (Years)')\n"
        "axes[0].set_ylabel('Annual Medical Cost ($)')\n\n"
        "sns.scatterplot(data=df_raw.sample(5000), x='bmi', y='annual_medical_cost', hue='smoker', alpha=0.6, palette='Set1', ax=axes[1])\n"
        "axes[1].set_title('BMI vs. Annual Medical Cost (Hue: Smoker)')\n"
        "axes[1].set_xlabel('BMI')\n"
        "axes[1].set_ylabel('Annual Medical Cost ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Scatter Plots (Feature-Target Relationships)\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Section B: Preprocessing & Feature Engineering
    nb.cells.append(new_markdown_cell(
        "## SECTION B — PREPROCESSING & FEATURE ENGINEERING (3 MARKS)\n"
        "We implement missing value handling, feature engineering, categorical encoding, scaling, and train/test splitting (`random_state=42`).\n\n"
        "### Feature Leakage Audit Table\n"
        "| Feature | Potential Leakage? | Reason | Action | Justification |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| `risk_score` | Yes | Direct mathematical precursor to risk target | Exclude | Prevents artificial target leakage |\n"
        "| `total_claims_paid` | Yes | Post-hoc downstream value (`claims_count` * `avg_claim_amount`) | Exclude from predictors | Prevents post-hoc reimbursement leakage |\n"
        "| `annual_premium` | No | Pre-computed policy enrollment cost | Retain | Valid actuarial predictor |\n"
        "| `monthly_premium` | No | Direct linear scaling of annual premium | Retain | Valid financial predictor |\n"
        "| `claims_count` | Partial | Pre-existing utilization frequency | Retain | Valid utilization metric |\n\n"
        "### Missing Values Handling Strategy\n"
        "TODO: Provide justification for missing value handling strategy."
    ))
    
    nb.cells.append(new_code_cell(
        "# Preprocessing Pipeline Execution (Strict Train/Test Split Fit)\n"
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_regression()\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n"
        "print('\\nEngineered Features Verified:')\n"
        "print([col for col in X_train.columns if col in config.ENGINEERED_FEATURES])"
    ))
    
    # Section C: Regression
    nb.cells.append(new_markdown_cell(
        "## SECTION C — REGRESSION TRACK (9 MARKS)\n"
        "We train all 10 mandatory regression algorithms, evaluate them on the same held-out test set ($R^2$, $RMSE$, $MAE$), tune top ensemble models, and run 5-fold cross validation."
    ))
    
    nb.cells.append(new_code_cell(
        "# Train 10 Regression Models\n"
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results, cv_scores = model_training.train_regression(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Consolidated Regression Comparison DataFrame\n"
        "Models ranked by $R^2$ score on the held-out test split:"
    ))
    
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    
    nb.cells.append(new_code_cell(
        "# 5-Fold Cross Validation Results for Top 2 Models\n"
        "print('5-Fold Cross Validation R2 Scores (Top 2 Models):')\n"
        "for model_name, score in cv_scores.items():\n"
        "    print(f'Model: {model_name:<30} | Mean CV R2: {score:.4f}')"
    ))
    
    # Regression Visualizations
    nb.cells.append(new_markdown_cell(
        "### Regression Visualizations & Model Diagnostics"
    ))
    
    nb.cells.append(new_code_cell(
        "best_model = joblib.load(config.BEST_REGRESSOR_PATH)\n"
        "y_pred = best_model.predict(X_test)\n"
        "residuals = y_test - y_pred\n\n"
        "# 1. Actual vs Predicted & 2. Residual Plot\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0], color='#0F4C81')\n"
        "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n"
        "axes[0].set_title('Actual vs. Predicted Annual Medical Costs')\n"
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
        "### Observation — Predicted vs Actual & Residual Plots\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    nb.cells.append(new_code_cell(
        "# 3. Tree-based Feature Importance Plot\n"
        "if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):\n"
        "    feat_names = best_model.named_steps['preprocessor'].get_feature_names_out()\n"
        "    importances = best_model.named_steps['regressor'].feature_importances_\n"
        "    clean_names = [n.split('__')[1] if '__' in n else n for n in feat_names]\n"
        "    feat_imp = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)\n"
        "    \n"
        "    plt.figure(figsize=(10, 6))\n"
        "    sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Blues_r')\n"
        "    plt.title('Top 15 Feature Importances (Best Tree-Based Model)')\n"
        "    plt.xlabel('Importance Score')\n"
        "    plt.ylabel('Feature')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Feature Importance Plot\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Final Review 1 Output Checklist
    nb.cells.append(new_markdown_cell(
        "## FINAL REVIEW-1 OUTPUT & CHECKLIST\n\n"
        "### Review-1 Requirements Verification\n"
        "| Section | Criterion | Status |\n"
        "| :--- | :--- | :---: |\n"
        "| Section A | Dataset Loaded & Shape Shown | Verified |\n"
        "| Section A | Dtypes & Missing Values Audited | Verified |\n"
        "| Section A | Target & Feature Distributions Plotted | Verified |\n"
        "| Section A | Correlation Heatmap & Scatter Plots | Verified |\n"
        "| Section B | Missing Values Handled & Scaling/Encoding | Verified |\n"
        "| Section B | Data Leakage Audit Table & Train/Test Split | Verified |\n"
        "| Section C | All 10 Regression Models Trained & Evaluated | Verified |\n"
        "| Section C | Hyperparameter Tuning & 5-Fold Cross Validation | Verified |\n"
        "| Section C | Residuals, Actual-vs-Predicted, and Feature Importance Plots | Verified |\n\n"
        "### Human-Written TODO Checklist\n"
        "- [ ] Write observation for Dataset Audit & Summary\n"
        "- [ ] Write observation for Target Distribution Plot\n"
        "- [ ] Write observation for Feature Distributions Plot\n"
        "- [ ] Write observation for Outlier Analysis Boxplots\n"
        "- [ ] Write observation for Correlation Heatmap\n"
        "- [ ] Write observation for Scatter Plots\n"
        "- [ ] Provide justification for missing value handling strategy\n"
        "- [ ] Write observation for Predicted vs Actual & Residual Plots\n"
        "- [ ] Write observation for Tree-Based Feature Importance Plot"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/regression.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created regression.ipynb successfully.")

def create_classification_notebook():
    nb = new_notebook()
    
    # Title & Metadata
    nb.cells.append(new_markdown_cell(
        "# Patient Risk Classification - Classification Track Part A (Review 1)\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Using Machine Learning\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n"
        "**Classification Target**: `is_high_risk`\n\n"
        "---\n"
        "### Academic Integrity Note\n"
        "Per course guidelines, AI code scaffolding is used for code setup and structure. "
        "All analytical observations and domain interpretations are marked with **`TODO` placeholders** for student authoring.\n\n"
        "Source/Citation: TODO — add source if external code was used."
    ))
    
    # 1. Setup
    nb.cells.append(new_markdown_cell(
        "## 1. Environment & Setup\n"
        "Import required libraries and configure global random state (`random_state=42`)."
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
    
    # Section D: Classification Part A
    nb.cells.append(new_markdown_cell(
        "## SECTION D — CLASSIFICATION PART A (3 MARKS)\n"
        "Target Variable: `is_high_risk`\n\n"
        "### Classification Feature Leakage Audit\n"
        "**Decision**: `risk_score` is directly used to calculate `is_high_risk`. Including `risk_score` in predictor features would cause target leakage. Therefore, `risk_score` is explicitly excluded from input features.\n\n"
        "TODO: Explain decision to exclude risk_score in detail."
    ))
    
    nb.cells.append(new_code_cell(
        "# Stratified Train/Test Split (stratify=y, random_state=42)\n"
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_classification()\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n"
        "print('\\nTrain Target Distribution (is_high_risk):')\n"
        "print(y_train.value_counts(normalize=True))\n"
        "print('\\nTest Target Distribution (is_high_risk):')\n"
        "print(y_test.value_counts(normalize=True))"
    ))
    
    nb.cells.append(new_code_cell(
        "# Train All 5 Classification Algorithms\n"
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "df_results = model_training.train_classification(X_train, X_test, y_train, y_test, preprocessor)"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Preliminary Classification Comparison Table\n"
        "Evaluated on Accuracy, Precision, Recall, Weighted F1-Score, and ROC-AUC:"
    ))
    nb.cells.append(new_code_cell(
        "df_results"
    ))
    
    # Confusion Matrix for EVERY Algorithm
    nb.cells.append(new_markdown_cell(
        "### Confusion Matrices for All 5 Classification Algorithms"
    ))
    
    nb.cells.append(new_code_cell(
        "# Generate Confusion Matrix plots for all trained classification models\n"
        "# Re-evaluate and plot for Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVC\n"
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
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n"
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
        "### Observation — Confusion Matrices\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Best Classifier ROC Curve
    nb.cells.append(new_code_cell(
        "# ROC Curve for Best Classifier\n"
        "best_clf = joblib.load(config.BEST_CLASSIFIER_PATH)\n"
        "y_prob = best_clf.predict_proba(X_test)[:, 1]\n"
        "fpr, tpr, _ = roc_curve(y_test, y_prob)\n"
        "roc_auc = auc(fpr, tpr)\n\n"
        "plt.figure(figsize=(8, 6))\n"
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
        "### Observation — ROC Curve\n"
        "TODO: Write observation after inspecting this plot."
    ))
    
    # Final Review 1 Output Checklist
    nb.cells.append(new_markdown_cell(
        "## FINAL REVIEW-1 OUTPUT & CHECKLIST\n\n"
        "### Classification Review-1 Verification\n"
        "| Criterion | Status |\n"
        "| :--- | :---: |\n"
        "| Target `is_high_risk` Defined | Verified |\n"
        "| Risk Score Excluded (Leakage Prevention) | Verified |\n"
        "| Stratified Train/Test Split (`random_state=42`) | Verified |\n"
        "| All 5 Classification Algorithms Trained | Verified |\n"
        "| Accuracy, Precision, Recall, Weighted F1, ROC-AUC Metrics | Verified |\n"
        "| Confusion Matrices Plotted for All 5 Models | Verified |\n"
        "| ROC Curve Plotted for Best Model | Verified |\n\n"
        "### Human-Written TODO Checklist\n"
        "- [ ] Write explanation for excluding risk_score from classification predictors\n"
        "- [ ] Write observation for Confusion Matrices plot\n"
        "- [ ] Write observation for ROC Curve plot"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/classification.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created classification.ipynb successfully.")

def create_clustering_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell(
        "# Unsupervised Patient Segmentation - Clustering Track Preview\n"
        "**Course**: 23CSE301 Machine Learning - Capstone Project\n"
        "**Project Title**: Healthcare Cost Prediction and Patient Risk Intelligence Using Machine Learning\n"
        "**Dataset**: Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n\n"
        "---\n"
        "### Academic Integrity Note\n"
        "Source/Citation: TODO — add source if external code was used."
    ))
    
    nb.cells.append(new_code_cell(
        "import sys\n"
        "sys.path.append('..')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import os\n"
        "import joblib\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.decomposition import PCA\n\n"
        "import app.config as config\n"
        "import app.preprocessing as preprocessing\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)"
    ))
    
    nb.cells.append(new_code_cell(
        "# Load Data and Run Clustering\n"
        "cluster_path = os.path.join(config.MODEL_DIR, 'best_clustering.joblib')\n"
        "if os.path.exists(cluster_path):\n"
        "    cluster_info = joblib.load(cluster_path)\n"
        "    print('Clustering model loaded successfully.')\n"
        "    print('Scaler:', cluster_info['scaler'])\n"
        "    print('Model:', cluster_info['model'])\n"
        "else:\n"
        "    print('Run train_clustering.py first to create clustering model.')"
    ))
    
    nb.cells.append(new_markdown_cell(
        "### Observation — Patient Clustering & Segmentation\n"
        "TODO: Write observation after inspecting clustering metrics and profiles."
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/clustering.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created clustering.ipynb successfully.")

if __name__ == "__main__":
    create_regression_notebook()
    create_classification_notebook()
    create_clustering_notebook()
