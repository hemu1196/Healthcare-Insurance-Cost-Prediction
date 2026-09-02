import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_regression_notebook():
    nb = new_notebook()
    
    # 01 Project Introduction & Header
    nb.cells.append(new_markdown_cell(
        "# 23CSE301 Machine Learning – Capstone Project\n\n"
        "**Team No:** 8\n\n"
        "**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform\n\n"
        "**Dataset:** Medical Insurance Cost Prediction Dataset (`medical_insurance.csv`)\n\n"
        "**Target Variable:** `annual_medical_cost` (Regression Track)\n\n"
        "### Project Introduction\n"
        "Predicting patient healthcare costs helps insurance providers estimate liabilities and plan resources. "
        "In this notebook, we implement and compare all 10 required regression algorithms on held-out test data."
    ))
    
    # 02 Import Libraries
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
        "import app.utils as utils\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n"
        "from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet\n"
        "from sklearn.tree import DecisionTreeRegressor\n"
        "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
        "from sklearn.svm import SVR\n"
        "from sklearn.neighbors import KNeighborsRegressor\n"
        "from sklearn.model_selection import RandomizedSearchCV, cross_val_score\n"
        "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)\n"
        "print('Libraries imported successfully. Random seed set to 42.')"
    ))
    
    # 03 Load Dataset & Audit
    nb.cells.append(new_markdown_cell(
        "## Load Dataset and Dataset Audit\n"
        "We load `medical_insurance.csv` and inspect dimensions, dtypes, missing values, and duplicates."
    ))
    nb.cells.append(new_code_cell(
        "df_raw = pd.read_csv(config.DATA_PATH)\n"
        "print('Dataset Shape (Rows, Columns):', df_raw.shape)\n"
        "print('Missing Values:', df_raw.isnull().sum().sum())\n"
        "print('Duplicates:', df_raw.duplicated().sum())\n"
        "df_raw.head()"
    ))
    
    # 04 Data Preprocessing & Train Test Split
    nb.cells.append(new_markdown_cell(
        "## Data Preprocessing and Train Test Split\n"
        "We split data into 80% train and 20% test splits (`random_state=42`). Preprocessing is fitted strictly on `X_train`."
    ))
    nb.cells.append(new_code_cell(
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_regression()\n"
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n\n"
        "regression_results = []"
    ))
    
    # 05 Individual Regression Models
    nb.cells.append(new_markdown_cell(
        "## Regression Models\n"
        "We train and evaluate each of the 10 required regression algorithms individually."
    ))
    
    # 1. Linear Regression
    nb.cells.append(new_markdown_cell("### 1. Linear Regression"))
    nb.cells.append(new_code_cell(
        "m_lr = Pipeline([('prep', preprocessor), ('reg', LinearRegression())])\n"
        "m_lr.fit(X_train, y_train)\n"
        "p_lr = m_lr.predict(X_test)\n"
        "r2_lr = r2_score(y_test, p_lr)\n"
        "rmse_lr = np.sqrt(mean_squared_error(y_test, p_lr))\n"
        "mae_lr = mean_absolute_error(y_test, p_lr)\n"
        "regression_results.append({'Model': 'Linear Regression', 'R2': r2_lr, 'RMSE': rmse_lr, 'MAE': mae_lr})\n"
        "print(f'Linear Regression -> R2: {r2_lr:.4f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}')"
    ))
    
    # 2. Ridge Regression
    nb.cells.append(new_markdown_cell("### 2. Ridge Regression"))
    nb.cells.append(new_code_cell(
        "m_ridge = Pipeline([('prep', preprocessor), ('reg', Ridge(alpha=1.0, random_state=42))])\n"
        "m_ridge.fit(X_train, y_train)\n"
        "p_ridge = m_ridge.predict(X_test)\n"
        "r2_ridge = r2_score(y_test, p_ridge)\n"
        "rmse_ridge = np.sqrt(mean_squared_error(y_test, p_ridge))\n"
        "mae_ridge = mean_absolute_error(y_test, p_ridge)\n"
        "regression_results.append({'Model': 'Ridge Regression', 'R2': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge})\n"
        "print(f'Ridge Regression -> R2: {r2_ridge:.4f}, RMSE: {rmse_ridge:.2f}, MAE: {mae_ridge:.2f}')"
    ))
    
    # 3. Lasso Regression
    nb.cells.append(new_markdown_cell("### 3. Lasso Regression"))
    nb.cells.append(new_code_cell(
        "m_lasso = Pipeline([('prep', preprocessor), ('reg', Lasso(alpha=1.0, max_iter=2000, random_state=42))])\n"
        "m_lasso.fit(X_train, y_train)\n"
        "p_lasso = m_lasso.predict(X_test)\n"
        "r2_lasso = r2_score(y_test, p_lasso)\n"
        "rmse_lasso = np.sqrt(mean_squared_error(y_test, p_lasso))\n"
        "mae_lasso = mean_absolute_error(y_test, p_lasso)\n"
        "regression_results.append({'Model': 'Lasso Regression', 'R2': r2_lasso, 'RMSE': rmse_lasso, 'MAE': mae_lasso})\n"
        "print(f'Lasso Regression -> R2: {r2_lasso:.4f}, RMSE: {rmse_lasso:.2f}, MAE: {mae_lasso:.2f}')"
    ))
    
    # 4. ElasticNet Regression
    nb.cells.append(new_markdown_cell("### 4. ElasticNet Regression"))
    nb.cells.append(new_code_cell(
        "m_enet = Pipeline([('prep', preprocessor), ('reg', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000, random_state=42))])\n"
        "m_enet.fit(X_train, y_train)\n"
        "p_enet = m_enet.predict(X_test)\n"
        "r2_enet = r2_score(y_test, p_enet)\n"
        "rmse_enet = np.sqrt(mean_squared_error(y_test, p_enet))\n"
        "mae_enet = mean_absolute_error(y_test, p_enet)\n"
        "regression_results.append({'Model': 'ElasticNet', 'R2': r2_enet, 'RMSE': rmse_enet, 'MAE': mae_enet})\n"
        "print(f'ElasticNet -> R2: {r2_enet:.4f}, RMSE: {rmse_enet:.2f}, MAE: {mae_enet:.2f}')"
    ))
    
    # 5. Polynomial Regression
    nb.cells.append(new_markdown_cell("### 5. Polynomial Regression"))
    nb.cells.append(new_code_cell(
        "m_poly = Pipeline([('prep', preprocessor), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('reg', LinearRegression())])\n"
        "m_poly.fit(X_train, y_train)\n"
        "p_poly = m_poly.predict(X_test)\n"
        "r2_poly = r2_score(y_test, p_poly)\n"
        "rmse_poly = np.sqrt(mean_squared_error(y_test, p_poly))\n"
        "mae_poly = mean_absolute_error(y_test, p_poly)\n"
        "regression_results.append({'Model': 'Polynomial Regression', 'R2': r2_poly, 'RMSE': rmse_poly, 'MAE': mae_poly})\n"
        "print(f'Polynomial Regression -> R2: {r2_poly:.4f}, RMSE: {rmse_poly:.2f}, MAE: {mae_poly:.2f}')"
    ))
    
    # 6. Decision Tree Regressor
    nb.cells.append(new_markdown_cell("### 6. Decision Tree Regressor"))
    nb.cells.append(new_code_cell(
        "m_dt = Pipeline([('prep', preprocessor), ('reg', DecisionTreeRegressor(max_depth=8, random_state=42))])\n"
        "m_dt.fit(X_train, y_train)\n"
        "p_dt = m_dt.predict(X_test)\n"
        "r2_dt = r2_score(y_test, p_dt)\n"
        "rmse_dt = np.sqrt(mean_squared_error(y_test, p_dt))\n"
        "mae_dt = mean_absolute_error(y_test, p_dt)\n"
        "regression_results.append({'Model': 'Decision Tree Regressor', 'R2': r2_dt, 'RMSE': rmse_dt, 'MAE': mae_dt})\n"
        "print(f'Decision Tree Regressor -> R2: {r2_dt:.4f}, RMSE: {rmse_dt:.2f}, MAE: {mae_dt:.2f}')"
    ))
    
    # 7. Random Forest Regressor
    nb.cells.append(new_markdown_cell("### 7. Random Forest Regressor"))
    nb.cells.append(new_code_cell(
        "m_rf = Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))])\n"
        "m_rf.fit(X_train, y_train)\n"
        "p_rf = m_rf.predict(X_test)\n"
        "r2_rf = r2_score(y_test, p_rf)\n"
        "rmse_rf = np.sqrt(mean_squared_error(y_test, p_rf))\n"
        "mae_rf = mean_absolute_error(y_test, p_rf)\n"
        "regression_results.append({'Model': 'Random Forest Regressor', 'R2': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf})\n"
        "print(f'Random Forest Regressor -> R2: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}')"
    ))
    
    # 8. Gradient Boosting Regressor
    nb.cells.append(new_markdown_cell("### 8. Gradient Boosting Regressor"))
    nb.cells.append(new_code_cell(
        "m_gb = Pipeline([('prep', preprocessor), ('reg', GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42))])\n"
        "m_gb.fit(X_train, y_train)\n"
        "p_gb = m_gb.predict(X_test)\n"
        "r2_gb = r2_score(y_test, p_gb)\n"
        "rmse_gb = np.sqrt(mean_squared_error(y_test, p_gb))\n"
        "mae_gb = mean_absolute_error(y_test, p_gb)\n"
        "regression_results.append({'Model': 'Gradient Boosting Regressor', 'R2': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb})\n"
        "print(f'Gradient Boosting Regressor -> R2: {r2_gb:.4f}, RMSE: {rmse_gb:.2f}, MAE: {mae_gb:.2f}')"
    ))
    
    # 9. Support Vector Regressor
    nb.cells.append(new_markdown_cell("### 9. Support Vector Regressor"))
    nb.cells.append(new_code_cell(
        "subsample_idx = np.random.choice(len(X_train), size=min(5000, len(X_train)), replace=False)\n"
        "m_svr = Pipeline([('prep', preprocessor), ('reg', SVR(C=1000.0, epsilon=0.1))])\n"
        "m_svr.fit(X_train.iloc[subsample_idx], y_train.iloc[subsample_idx])\n"
        "p_svr = m_svr.predict(X_test)\n"
        "r2_svr = r2_score(y_test, p_svr)\n"
        "rmse_svr = np.sqrt(mean_squared_error(y_test, p_svr))\n"
        "mae_svr = mean_absolute_error(y_test, p_svr)\n"
        "regression_results.append({'Model': 'Support Vector Regressor', 'R2': r2_svr, 'RMSE': rmse_svr, 'MAE': mae_svr})\n"
        "print(f'Support Vector Regressor -> R2: {r2_svr:.4f}, RMSE: {rmse_svr:.2f}, MAE: {mae_svr:.2f}')"
    ))
    
    # 10. KNN Regressor
    nb.cells.append(new_markdown_cell("### 10. KNN Regressor"))
    nb.cells.append(new_code_cell(
        "m_knn = Pipeline([('prep', preprocessor), ('reg', KNeighborsRegressor(n_neighbors=7, n_jobs=-1))])\n"
        "m_knn.fit(X_train.iloc[subsample_idx], y_train.iloc[subsample_idx])\n"
        "p_knn = m_knn.predict(X_test)\n"
        "r2_knn = r2_score(y_test, p_knn)\n"
        "rmse_knn = np.sqrt(mean_squared_error(y_test, p_knn))\n"
        "mae_knn = mean_absolute_error(y_test, p_knn)\n"
        "regression_results.append({'Model': 'KNN Regressor', 'R2': r2_knn, 'RMSE': rmse_knn, 'MAE': mae_knn})\n"
        "print(f'KNN Regressor -> R2: {r2_knn:.4f}, RMSE: {rmse_knn:.2f}, MAE: {mae_knn:.2f}')"
    ))
    
    # Regression Model Comparison
    nb.cells.append(new_markdown_cell(
        "## Regression Model Comparison\n"
        "Consolidated DataFrame comparing all 10 regression models, ranked by R² score in descending order."
    ))
    nb.cells.append(new_code_cell(
        "df_reg = pd.DataFrame(regression_results).sort_values(by='R2', ascending=False).reset_index(drop=True)\n"
        "df_reg.index += 1\n"
        "df_reg"
    ))
    
    # Hyperparameter Tuning & CV & Visualizations
    nb.cells.append(new_markdown_cell(
        "## Hyperparameter Tuning, Cross Validation & Visualizations\n"
        "We tune Random Forest using `RandomizedSearchCV`, compute 5-fold cross-validation R² scores, and plot diagnostics."
    ))
    nb.cells.append(new_code_cell(
        "rf_param_grid = {'reg__n_estimators': [50, 100], 'reg__max_depth': [10, 15, None], 'reg__min_samples_split': [2, 5]}\n"
        "search = RandomizedSearchCV(m_rf, rf_param_grid, n_iter=4, cv=3, scoring='r2', random_state=42, n_jobs=-1)\n"
        "search.fit(X_train.iloc[:10000], y_train.iloc[:10000])\n"
        "best_rf_model = search.best_estimator_\n"
        "print('Best RF Hyperparameters:', search.best_params_)\n\n"
        "cv_scores = cross_val_score(best_rf_model, X_train.iloc[:10000], y_train.iloc[:10000], cv=5, scoring='r2')\n"
        "print(f'5-Fold CV Mean R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')"
    ))
    
    nb.cells.append(new_code_cell(
        "y_pred_best = best_rf_model.predict(X_test)\n"
        "residuals = y_test - y_pred_best\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
        "sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.5, ax=axes[0], color='#0F4C81')\n"
        "axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n"
        "axes[0].set_title('Actual vs. Predicted Medical Cost')\n"
        "axes[0].set_xlabel('Actual Cost ($)')\n"
        "axes[0].set_ylabel('Predicted Cost ($)')\n\n"
        "sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.5, ax=axes[1], color='#E74C3C')\n"
        "axes[1].axhline(y=0, color='black', linestyle='--', lw=2)\n"
        "axes[1].set_title('Residuals vs. Predicted Values')\n"
        "axes[1].set_xlabel('Predicted Cost ($)')\n"
        "axes[1].set_ylabel('Residual ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/regression.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created regression.ipynb with 10 individual algorithm cells.")

def create_classification_notebook():
    nb = new_notebook()
    
    # 01 Introduction & Setup
    nb.cells.append(new_markdown_cell(
        "# 23CSE301 Machine Learning – Capstone Project\n\n"
        "**Team No:** 8\n\n"
        "**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform\n\n"
        "**Target Variable:** `is_high_risk` (Classification Part A)\n\n"
        "### Project Introduction\n"
        "We train and evaluate all 5 required classification models individually on stratified train/test data."
    ))
    
    # 02 Import Libraries
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
        "import app.utils as utils\n\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.naive_bayes import GaussianNB\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.svm import SVC\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "from sklearn.base import TransformerMixin, BaseEstimator\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc\n\n"
        "class DenseTransformer(TransformerMixin, BaseEstimator):\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n"
        "    def transform(self, X, y=None):\n"
        "        if hasattr(X, 'toarray'):\n"
        "            return X.toarray()\n"
        "        return X\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)\n"
        "print('Classification libraries loaded successfully.')"
    ))
    
    # 03 Preprocessing & Stratified Split
    nb.cells.append(new_markdown_cell(
        "## Data Preprocessing and Stratified Split\n"
        "Target variable `is_high_risk`. `risk_score` is dropped to prevent target leakage."
    ))
    nb.cells.append(new_code_cell(
        "X_train, X_test, y_train, y_test, cat_cols, num_cols, bin_cols = preprocessing.prepare_data_classification()\n"
        "preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)\n"
        "print('Train Set Shape:', X_train.shape)\n"
        "print('Test Set Shape:', X_test.shape)\n\n"
        "cls_results = []\n"
        "subsample_idx = np.random.choice(len(X_train), size=min(8000, len(X_train)), replace=False)\n"
        "X_train_sub = X_train.iloc[subsample_idx]\n"
        "y_train_sub = y_train.iloc[subsample_idx]"
    ))
    
    # 5 Individual Classification Models
    # 1. Logistic Regression
    nb.cells.append(new_markdown_cell("### 1. Logistic Regression"))
    nb.cells.append(new_code_cell(
        "c_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])\n"
        "c_lr.fit(X_train, y_train)\n"
        "p_lr = c_lr.predict(X_test)\n"
        "prob_lr = c_lr.predict_proba(X_test)[:, 1]\n"
        "cls_results.append({\n"
        "    'Model': 'Logistic Regression',\n"
        "    'Accuracy': accuracy_score(y_test, p_lr),\n"
        "    'Precision': precision_score(y_test, p_lr, zero_division=0),\n"
        "    'Recall': recall_score(y_test, p_lr, zero_division=0),\n"
        "    'Weighted F1': f1_score(y_test, p_lr, average='weighted'),\n"
        "    'ROC-AUC': roc_auc_score(y_test, prob_lr)\n"
        "})\n"
        "print('Logistic Regression complete.')"
    ))
    
    # 2. KNN Classifier
    nb.cells.append(new_markdown_cell("### 2. K-Nearest Neighbors Classifier"))
    nb.cells.append(new_code_cell(
        "c_knn = Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=7, n_jobs=-1))])\n"
        "c_knn.fit(X_train_sub, y_train_sub)\n"
        "p_knn = c_knn.predict(X_test)\n"
        "prob_knn = c_knn.predict_proba(X_test)[:, 1]\n"
        "cls_results.append({\n"
        "    'Model': 'K-Nearest Neighbors',\n"
        "    'Accuracy': accuracy_score(y_test, p_knn),\n"
        "    'Precision': precision_score(y_test, p_knn, zero_division=0),\n"
        "    'Recall': recall_score(y_test, p_knn, zero_division=0),\n"
        "    'Weighted F1': f1_score(y_test, p_knn, average='weighted'),\n"
        "    'ROC-AUC': roc_auc_score(y_test, prob_knn)\n"
        "})\n"
        "print('KNN Classifier complete.')"
    ))
    
    # 3. Gaussian Naive Bayes
    nb.cells.append(new_markdown_cell("### 3. Gaussian Naive Bayes"))
    nb.cells.append(new_code_cell(
        "c_gnb = Pipeline([('prep', preprocessor), ('to_dense', DenseTransformer()), ('clf', GaussianNB())])\n"
        "c_gnb.fit(X_train, y_train)\n"
        "p_gnb = c_gnb.predict(X_test)\n"
        "prob_gnb = c_gnb.predict_proba(X_test)[:, 1]\n"
        "cls_results.append({\n"
        "    'Model': 'Gaussian Naive Bayes',\n"
        "    'Accuracy': accuracy_score(y_test, p_gnb),\n"
        "    'Precision': precision_score(y_test, p_gnb, zero_division=0),\n"
        "    'Recall': recall_score(y_test, p_gnb, zero_division=0),\n"
        "    'Weighted F1': f1_score(y_test, p_gnb, average='weighted'),\n"
        "    'ROC-AUC': roc_auc_score(y_test, prob_gnb)\n"
        "})\n"
        "print('Gaussian Naive Bayes complete.')"
    ))
    
    # 4. Decision Tree Classifier
    nb.cells.append(new_markdown_cell("### 4. Decision Tree Classifier"))
    nb.cells.append(new_code_cell(
        "c_dt = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))])\n"
        "c_dt.fit(X_train, y_train)\n"
        "p_dt = c_dt.predict(X_test)\n"
        "prob_dt = c_dt.predict_proba(X_test)[:, 1]\n"
        "cls_results.append({\n"
        "    'Model': 'Decision Tree',\n"
        "    'Accuracy': accuracy_score(y_test, p_dt),\n"
        "    'Precision': precision_score(y_test, p_dt, zero_division=0),\n"
        "    'Recall': recall_score(y_test, p_dt, zero_division=0),\n"
        "    'Weighted F1': f1_score(y_test, p_dt, average='weighted'),\n"
        "    'ROC-AUC': roc_auc_score(y_test, prob_dt)\n"
        "})\n"
        "print('Decision Tree Classifier complete.')"
    ))
    
    # 5. Support Vector Classifier
    nb.cells.append(new_markdown_cell("### 5. Support Vector Classifier"))
    nb.cells.append(new_code_cell(
        "c_svc = Pipeline([('prep', preprocessor), ('clf', CalibratedClassifierCV(SVC(C=1.0, random_state=42), ensemble=False))])\n"
        "c_svc.fit(X_train_sub, y_train_sub)\n"
        "p_svc = c_svc.predict(X_test)\n"
        "prob_svc = c_svc.predict_proba(X_test)[:, 1]\n"
        "cls_results.append({\n"
        "    'Model': 'Support Vector Machine',\n"
        "    'Accuracy': accuracy_score(y_test, p_svc),\n"
        "    'Precision': precision_score(y_test, p_svc, zero_division=0),\n"
        "    'Recall': recall_score(y_test, p_svc, zero_division=0),\n"
        "    'Weighted F1': f1_score(y_test, p_svc, average='weighted'),\n"
        "    'ROC-AUC': roc_auc_score(y_test, prob_svc)\n"
        "})\n"
        "print('Support Vector Classifier complete.')"
    ))
    
    # Classification Comparison & Confusion Matrices
    nb.cells.append(new_markdown_cell(
        "## Classification Model Comparison\n"
        "Consolidated metrics for all 5 classification algorithms:"
    ))
    nb.cells.append(new_code_cell(
        "df_cls = pd.DataFrame(cls_results).sort_values(by='Weighted F1', ascending=False).reset_index(drop=True)\n"
        "df_cls.index += 1\n"
        "df_cls"
    ))
    
    nb.cells.append(new_markdown_cell("## Confusion Matrices and ROC Curve"))
    nb.cells.append(new_code_cell(
        "all_models = {'Logistic Regression': c_lr, 'KNN Classifier': c_knn, 'Gaussian Naive Bayes': c_gnb, 'Decision Tree': c_dt, 'Support Vector Classifier': c_svc}\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n"
        "axes = axes.flatten()\n"
        "for idx, (name, model) in enumerate(all_models.items()):\n"
        "    preds = model.predict(X_test)\n"
        "    cm = confusion_matrix(y_test, preds)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=['Low/Med', 'High'], yticklabels=['Low/Med', 'High'])\n"
        "    axes[idx].set_title(f'Confusion Matrix: {name}')\n"
        "    axes[idx].set_xlabel('Predicted')\n"
        "    axes[idx].set_ylabel('Actual')\n"
        "axes[5].axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb.cells.append(new_code_cell(
        "fpr, tpr, _ = roc_curve(y_test, prob_dt)\n"
        "roc_auc = auc(fpr, tpr)\n"
        "plt.figure(figsize=(7, 5))\n"
        "plt.plot(fpr, tpr, color='#0F4C81', lw=2, label=f'Decision Tree (AUC = {roc_auc:.4f})')\n"
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n"
        "plt.title('ROC Curve - Top Classifier')\n"
        "plt.xlabel('False Positive Rate')\n"
        "plt.ylabel('True Positive Rate')\n"
        "plt.legend(loc='lower right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/classification.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created classification.ipynb with 5 individual classifier cells.")

def create_clustering_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell(
        "# 23CSE301 Machine Learning – Capstone Project\n\n"
        "**Team No:** 8\n\n"
        "**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform\n\n"
        "**Module:** Unsupervised Patient Clustering & Segmentation\n\n"
        "### Overview\n"
        "We perform unsupervised clustering on patient clinical and utilization metrics using K-Means, Agglomerative, DBSCAN, Gaussian Mixture, and Spectral Clustering."
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
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering\n"
        "from sklearn.mixture import GaussianMixture\n"
        "from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.manifold import TSNE\n"
        "from scipy.cluster.hierarchy import dendrogram, linkage\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "np.random.seed(config.RANDOM_STATE)\n"
        "print('Clustering libraries loaded successfully.')"
    ))
    
    nb.cells.append(new_code_cell(
        "# Load dataset and preprocess for clustering\n"
        "df_raw = pd.read_csv(config.DATA_PATH)\n"
        "df_cluster = df_raw.drop(columns=[c for c in [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET, 'person_id', 'risk_score'] if c in df_raw.columns])\n\n"
        "# Handle missing values\n"
        "for col in df_cluster.select_dtypes(include='object').columns:\n"
        "    df_cluster[col] = df_cluster[col].fillna(df_cluster[col].mode()[0])\n"
        "for col in df_cluster.select_dtypes(include=np.number).columns:\n"
        "    df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())\n\n"
        "# One-hot encoding and scaling\n"
        "df_encoded = pd.get_dummies(df_cluster, drop_first=True)\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(df_encoded)\n"
        "print('Scaled clustering data shape:', X_scaled.shape)"
    ))
    
    nb.cells.append(new_code_cell(
        "# Fit clustering models on subsample for efficiency\n"
        "sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)\n"
        "X_sample = X_scaled[sample_idx]\n\n"
        "models = {\n"
        "    'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),\n"
        "    'Agglomerative': AgglomerativeClustering(n_clusters=4, linkage='ward'),\n"
        "    'DBSCAN': DBSCAN(eps=1.5, min_samples=5),\n"
        "    'Gaussian Mixture': GaussianMixture(n_components=4, random_state=42),\n"
        "    'Spectral Clustering': SpectralClustering(n_clusters=4, random_state=42, affinity='nearest_neighbors')\n"
        "}\n\n"
        "cluster_results = []\n"
        "for name, model in models.items():\n"
        "    labels = model.fit_predict(X_sample)\n"
        "    if len(set(labels)) > 1:\n"
        "        sil = silhouette_score(X_sample, labels)\n"
        "        db = davies_bouldin_score(X_sample, labels)\n"
        "        ch = calinski_harabasz_score(X_sample, labels)\n"
        "        cluster_results.append({'Algorithm': name, 'Silhouette Score': sil, 'Davies-Bouldin Index': db, 'Calinski-Harabasz Index': ch})\n"
        "    else:\n"
        "        print(f'{name} produced 1 cluster.')\n\n"
        "df_cluster_res = pd.DataFrame(cluster_results).sort_values(by='Silhouette Score', ascending=False).reset_index(drop=True)\n"
        "df_cluster_res"
    ))
    
    nb.cells.append(new_code_cell(
        "# PCA 2D Visualization & t-SNE Visualization\n"
        "kmeans_best = KMeans(n_clusters=4, random_state=42, n_init=10)\n"
        "labels_best = kmeans_best.fit_predict(X_sample)\n\n"
        "pca = PCA(n_components=2)\n"
        "X_pca = pca.fit_transform(X_sample)\n\n"
        "tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n"
        "X_tsne = tsne.fit_transform(X_sample[:1000])\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
        "sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_best, palette='tab10', ax=axes[0], alpha=0.7)\n"
        "axes[0].set_title('PCA 2D Cluster Projection')\n"
        "axes[0].set_xlabel('PCA Component 1')\n"
        "axes[0].set_ylabel('PCA Component 2')\n\n"
        "sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_best[:1000], palette='tab10', ax=axes[1], alpha=0.7)\n"
        "axes[1].set_title('t-SNE 2D Projection (subsample)')\n"
        "axes[1].set_xlabel('t-SNE 1')\n"
        "axes[1].set_ylabel('t-SNE 2')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/notebooks", exist_ok=True)
    with open("/Users/hemachandra/Documents/ML_capstone/notebooks/clustering.ipynb", "w") as f:
        nbformat.write(nb, f)
    print("Created clustering.ipynb successfully.")

def create_master_review_1_notebook():
    create_regression_notebook()
    create_classification_notebook()
    create_clustering_notebook()

if __name__ == "__main__":
    create_master_review_1_notebook()
