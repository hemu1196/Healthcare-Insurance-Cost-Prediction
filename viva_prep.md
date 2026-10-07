# 🎓 Faculty Viva & Technical Defense Preparation Guide

**Course:** 23CSE301 Machine Learning – Capstone Project  
**Team No:** 8  
**Project Title:** Healthcare Cost Prediction and Patient Risk Intelligence Platform  

---

## ❓ Frequently Asked Faculty Viva Questions & Technical Answers

### 1. Data Preprocessing & Leakage Prevention
- **Q: Why did you choose median imputation for missing numerical values and 'Unknown' for missing categoricals?**
  - **Answer:** Median imputation is robust to right-skewed medical distributions (such as hospital visit counts or claim amounts) compared to the mean, which is pulled by extreme outliers. For categorical features like `alcohol_freq`, imputing `'Unknown'` preserves non-response bias without inventing artificial patient habits.
- **Q: Why MUST scaling and encoding be fitted strictly on training data?**
  - **Answer:** Fitting scalers or encoders on the entire dataset before splitting causes **data leakage**. Information from the test set (such as mean and variance) leaks into the training pipeline, leading to overly optimistic test performance that fails in real-world deployment.
- **Q: How did you prevent target leakage in Regression and Classification?**
  - **Answer:** In regression, `total_claims_paid` represents post-hoc claim reimbursements calculated after medical costs occur; dropping it ensured we only used pre-outcome predictors. In classification, `risk_score` is a synthetic variable directly used to derive `is_high_risk`; excluding `risk_score` prevented 100% artificial target leakage.

### 2. Supervised Learning & Algorithm Comparison
- **Q: Why split classification into Part A and Part B?**
  - **Answer:** Part A evaluates baseline and probabilistic models (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC) for Review 1. Part B introduces advanced ensemble methods (Random Forest, AdaBoost, Gradient Boosting, Bagging) and Neural Networks (MLP Classifier) for Review 2 to evaluate complex non-linear feature interactions.
- **Q: Why is Weighted F1-score preferred over Accuracy for classification evaluation?**
  - **Answer:** Accuracy can be misleading if class distributions shift or when evaluating cost-sensitive domains. Weighted F1 accounts for class proportions while balancing Precision and Recall, making it ideal for clinical risk stratification.
- **Q: What is the clinical significance of False Negatives (FN)?**
  - **Answer:** A False Negative occurs when a high-risk patient is incorrectly classified as low risk. In healthcare, an FN leads to delayed preventative clinical interventions, resulting in preventable emergency hospital readmissions and high human/financial cost.

### 3. Hyperparameter Tuning & Cross-Validation
- **Q: What is the difference between k-Fold CV and Stratified k-Fold CV?**
  - **Answer:** Standard k-fold randomly splits data into $k$ folds, which might alter class proportions in small or imbalanced datasets. Stratified k-fold ensures that every fold maintains the exact class proportion (e.g. 50% high risk / 50% low risk) as the full dataset.
- **Q: Why use `RandomizedSearchCV` instead of `GridSearchCV`?**
  - **Answer:** `GridSearchCV` exhaustively tests every parameter combination, which is computationally expensive on 100,000 rows. `RandomizedSearchCV` samples a fixed number of parameter combinations across the search space, achieving comparable optimal parameters in a fraction of the compute time.

### 4. Dimensionality Reduction (PCA)
- **Q: Why was PCA applied, and why did you compare models with vs. without PCA?**
  - **Answer:** PCA transforms correlated features into orthogonal principal components. We retained 36 components to capture 95% dataset variance. However, tree-based models (Random Forest, Decision Tree) perform better on raw engineered features without PCA because orthogonal component combinations dilute direct clinical feature split interpretability.

### 5. Unsupervised Clustering
- **Q: Why can't ground-truth target labels be used during clustering model fitting?**
  - **Answer:** Clustering is an unsupervised task designed to discover natural patient groupings without supervision. Including class labels during training leads to supervised leakage and invalidates cluster validation metrics.
- **Q: How do Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index differ?**
  - **Answer:**
    - **Silhouette Score** ($[-1, +1]$): Measures cluster cohesion vs separation. Higher is better.
    - **Davies-Bouldin Index**: Measures average similarity between clusters. Lower is better.
    - **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.
- **Q: Why use PCA vs t-SNE for cluster visualization?**
  - **Answer:** PCA is a linear projection that preserves global dataset variance, making it fast and reproducible. t-SNE is a non-linear manifold technique that preserves local pairwise distances, creating distinct visual clusters in 2D space.
