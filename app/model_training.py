import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import app.config as config
import app.preprocessing as preprocessing

def train_regression(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate 10 regression algorithms.
    Tuning is applied to RF and Gradient Boosting.
    Top 2 models are evaluated using 5-fold CV.
    The best model is stored as best_regressor.joblib.
    """
    results = {}
    trained_models = {}
    
    # 1. Linear Regression
    lr = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    lr.fit(X_train, y_train)
    trained_models["Linear Regression"] = lr
    
    # 2. Ridge Regression
    ridge = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    trained_models["Ridge Regression"] = ridge
    
    # 3. Lasso Regression
    lasso = Pipeline([("preprocessor", preprocessor), ("regressor", Lasso(alpha=1.0, max_iter=2000))])
    lasso.fit(X_train, y_train)
    trained_models["Lasso Regression"] = lasso
    
    # 4. ElasticNet
    enet = Pipeline([("preprocessor", preprocessor), ("regressor", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000))])
    enet.fit(X_train, y_train)
    trained_models["ElasticNet"] = enet
    
    # 5. Polynomial Regression (Degree 2)
    # PolynomialFeatures on selected columns: age, bmi, visits_last_year
    poly_cols = ["age", "bmi", "visits_last_year"]
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    
    poly_transformer = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    poly_preprocessor = ColumnTransformer(
        transformers=[
            ("poly_num", poly_transformer, poly_cols),
            ("num", StandardScaler(), [c for c in X_train.columns if c not in poly_cols and c in config.NUMERICAL_FEATURES]),
            ("cat", preprocessor.named_transformers_["cat"], [c for c in X_train.columns if c in config.CATEGORICAL_FEATURES or c in ["BMI_Category", "Age_Group"]]),
            ("bin", "passthrough", [c for c in X_train.columns if c in config.BINARY_CLINICAL_FEATURES])
        ]
    )
    
    poly_reg = Pipeline([("preprocessor", poly_preprocessor), ("regressor", LinearRegression())])
    poly_reg.fit(X_train, y_train)
    trained_models["Polynomial Regression"] = poly_reg
    
    # 6. Decision Tree Regressor
    dt_reg = Pipeline([("preprocessor", preprocessor), ("regressor", DecisionTreeRegressor(max_depth=8, random_state=config.RANDOM_STATE))])
    dt_reg.fit(X_train, y_train)
    trained_models["Decision Tree Regressor"] = dt_reg
    
    # 7. Random Forest Regressor
    rf_reg = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=config.RANDOM_STATE, n_jobs=-1))])
    rf_reg.fit(X_train, y_train)
    trained_models["Random Forest Regressor"] = rf_reg
    
    # 8. Gradient Boosting Regressor
    gb_reg = Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=config.RANDOM_STATE))])
    gb_reg.fit(X_train, y_train)
    trained_models["Gradient Boosting Regressor"] = gb_reg
    
    # Subsample index for SVR and KNN
    subsample_idx = np.random.choice(len(X_train), size=min(8000, len(X_train)), replace=False)
    X_train_sub = X_train.iloc[subsample_idx]
    y_train_sub = y_train.iloc[subsample_idx]
    
    # 9. Support Vector Regressor (SVR)
    svr = Pipeline([("preprocessor", preprocessor), ("regressor", SVR(C=1000.0, epsilon=0.1))])
    print("Training SVR (on 8,000 row subsample for speed)...")
    svr.fit(X_train_sub, y_train_sub)
    trained_models["Support Vector Regressor"] = svr
    
    # 10. KNN Regressor
    knn_reg = Pipeline([("preprocessor", preprocessor), ("regressor", KNeighborsRegressor(n_neighbors=7, n_jobs=-1))])
    print("Training KNN Regressor (on 8,000 row subsample for speed)...")
    knn_reg.fit(X_train_sub, y_train_sub)
    trained_models["KNN Regressor"] = knn_reg
    
    # Evaluate baseline models
    print("\nEvaluating Baseline Regression Models:")
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"{name:<30} | R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
    # Hyperparameter Tuning on RF & Gradient Boosting
    print("\n--- Hyperparameter Tuning using RandomizedSearchCV ---")
    
    # RF Tuning
    print("Tuning Random Forest Regressor...")
    rf_grid = config.REG_TUNING_GRIDS["Random Forest"]
    rf_cv = RandomizedSearchCV(
        estimator=trained_models["Random Forest Regressor"], 
        param_distributions=rf_grid, 
        n_iter=4, 
        cv=3, 
        scoring="r2", 
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf_cv.fit(X_train_sub, y_train_sub)
    best_rf = rf_cv.best_estimator_
    print("Best RF Params:", rf_cv.best_params_)
    
    # GB Tuning
    print("Tuning Gradient Boosting Regressor...")
    gb_grid = config.REG_TUNING_GRIDS["Gradient Boosting"]
    gb_cv = RandomizedSearchCV(
        estimator=trained_models["Gradient Boosting Regressor"], 
        param_distributions=gb_grid, 
        n_iter=4, 
        cv=3, 
        scoring="r2", 
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    gb_cv.fit(X_train_sub, y_train_sub)
    best_gb = gb_cv.best_estimator_
    print("Best GB Params:", gb_cv.best_params_)
    
    # Re-evaluate Tuned Models
    print("Fitting best Random Forest on full train set...")
    best_rf.fit(X_train, y_train)
    rf_preds = best_rf.predict(X_test)
    results["Tuned Random Forest"] = {
        "R2": r2_score(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "MAE": mean_absolute_error(y_test, rf_preds)
    }
    trained_models["Tuned Random Forest"] = best_rf
    
    print("Fitting best Gradient Boosting on full train set...")
    best_gb.fit(X_train, y_train)
    gb_preds = best_gb.predict(X_test)
    results["Tuned Gradient Boosting"] = {
        "R2": r2_score(y_test, gb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, gb_preds)),
        "MAE": mean_absolute_error(y_test, gb_preds)
    }
    trained_models["Tuned Gradient Boosting"] = best_gb
    
    # 5-fold cross validation for top 2 models on train set
    top_models = ["Tuned Gradient Boosting", "Tuned Random Forest"]
    cv_scores = {}
    for name in top_models:
        model = trained_models[name]
        print(f"Calculating 5-Fold Cross-Validation R2 for {name} (on subsample)...")
        scores = cross_val_score(model, X_train_sub, y_train_sub, cv=5, scoring="r2", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"5-Fold CV R2 for {name}: {scores.mean():.4f}")
        
    # Rank models by R2
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.rename(columns={"index": "Model"}, inplace=True)
    df_results = df_results.sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    # Save results table
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    df_results.to_csv(config.REGRESSION_METRICS_PATH, index=False)
    
    # Save best model with compression to avoid GitHub file size limits
    best_model_name = df_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_REGRESSOR_PATH, compress=3)
    print(f"Best Regressor Model: {best_model_name} saved to {config.BEST_REGRESSOR_PATH}")
    
    return df_results, cv_scores

def train_classification(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate 5 classification algorithms (Review 1 Part A).
    Tuning is applied to Logistic Regression and Decision Tree.
    Best classifier is stored as best_classifier.joblib.
    """
    results = {}
    trained_models = {}
    
    # 1. Logistic Regression
    lr = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE))])
    lr.fit(X_train, y_train)
    trained_models["Logistic Regression"] = lr
    
    # Subsample for SVC and KNN speed
    subsample_idx = np.random.choice(len(X_train), size=min(8000, len(X_train)), replace=False)
    X_train_sub = X_train.iloc[subsample_idx]
    y_train_sub = y_train.iloc[subsample_idx]
    
    # 2. K-Nearest Neighbors (KNN)
    knn = Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))])
    print("Training KNN Classifier (on 8,000 row subsample for speed)...")
    knn.fit(X_train_sub, y_train_sub)
    trained_models["K-Nearest Neighbors"] = knn
    
    # 3. Naive Bayes (Gaussian)
    from sklearn.base import TransformerMixin, BaseEstimator
    class DenseTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            if hasattr(X, "toarray"):
                return X.toarray()
            return X
            
    nb = Pipeline([
        ("preprocessor", preprocessor), 
        ("to_dense", DenseTransformer()), 
        ("classifier", GaussianNB())
    ])
    nb.fit(X_train, y_train)
    trained_models["Naive Bayes"] = nb
    
    # 4. Decision Tree Classifier
    dt = Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(max_depth=8, random_state=config.RANDOM_STATE))])
    dt.fit(X_train, y_train)
    trained_models["Decision Tree"] = dt
    
    # 5. Support Vector Machine (SVC)
    svc = Pipeline([("preprocessor", preprocessor), ("classifier", SVC(probability=True, C=1.0, random_state=config.RANDOM_STATE))])
    print("Training Support Vector Machine (on 8,000 row subsample for speed)...")
    svc.fit(X_train_sub, y_train_sub)
    trained_models["Support Vector Machine"] = svc
    
    # Evaluate baseline models
    print("\nEvaluating Baseline Classification Models:")
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        print(f"{name:<30} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # Tuning Logistic Regression and Decision Tree
    print("\n--- Hyperparameter Tuning using RandomizedSearchCV ---")
    
    # Logistic Regression Tuning
    lr_grid = config.CLS_TUNING_GRIDS["Logistic Regression"]
    lr_cv = RandomizedSearchCV(
        estimator=trained_models["Logistic Regression"],
        param_distributions=lr_grid,
        n_iter=3,
        cv=3,
        scoring="accuracy",
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    lr_cv.fit(X_train_sub, y_train_sub)
    best_lr = lr_cv.best_estimator_
    print("Best Logistic Regression Params:", lr_cv.best_params_)
    
    # Decision Tree Tuning
    dt_grid = config.CLS_TUNING_GRIDS["Decision Tree"]
    dt_cv = RandomizedSearchCV(
        estimator=trained_models["Decision Tree"],
        param_distributions=dt_grid,
        n_iter=4,
        cv=3,
        scoring="accuracy",
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    dt_cv.fit(X_train_sub, y_train_sub)
    best_dt = dt_cv.best_estimator_
    print("Best Decision Tree Params:", dt_cv.best_params_)
    
    # Re-evaluate Tuned Models
    print("Fitting best Logistic Regression on full train set...")
    best_lr.fit(X_train, y_train)
    lr_preds = best_lr.predict(X_test)
    lr_probs = best_lr.predict_proba(X_test)[:, 1]
    results["Tuned Logistic Regression"] = {
        "Accuracy": accuracy_score(y_test, lr_preds),
        "Precision": precision_score(y_test, lr_preds, zero_division=0),
        "Recall": recall_score(y_test, lr_preds, zero_division=0),
        "F1-Score": f1_score(y_test, lr_preds, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, lr_probs)
    }
    trained_models["Tuned Logistic Regression"] = best_lr
    
    print("Fitting best Decision Tree on full train set...")
    best_dt.fit(X_train, y_train)
    dt_preds = best_dt.predict(X_test)
    dt_probs = best_dt.predict_proba(X_test)[:, 1]
    results["Tuned Decision Tree"] = {
        "Accuracy": accuracy_score(y_test, dt_preds),
        "Precision": precision_score(y_test, dt_preds, zero_division=0),
        "Recall": recall_score(y_test, dt_preds, zero_division=0),
        "F1-Score": f1_score(y_test, dt_preds, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, dt_probs)
    }
    trained_models["Tuned Decision Tree"] = best_dt
    
    # Rank models by F1-Score
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.rename(columns={"index": "Model"}, inplace=True)
    df_results = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    
    # Save results table
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    df_results.to_csv(config.CLASSIFICATION_METRICS_PATH, index=False)
    
    # Save best classifier with compression to avoid GitHub file size limits
    best_model_name = df_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_CLASSIFIER_PATH, compress=3)
    print(f"Best Classifier Model: {best_model_name} saved to {config.BEST_CLASSIFIER_PATH}")
    
    return df_results
