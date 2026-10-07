import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import app.config as config
import app.preprocessing as preprocessing

def train_clustering():
    print("Training clustering model for Patient Segmentation (Review 2 sneak-peek/bonus)...")
    
    # Load and preprocess
    df = pd.read_csv(config.DATA_PATH)
    df_eng = preprocessing.engineer_features(df)
    
    drop_cols = [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET, 'person_id', 'risk_score', 'total_claims_paid']
    X_raw = df_eng.drop(columns=[col for col in drop_cols if col in df_eng.columns], errors='ignore')
    
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in X_raw.columns] + ['BMI_Category', 'Age_Group']
    bin_cols = [c for c in config.BINARY_CLINICAL_FEATURES if c in X_raw.columns]
    num_cols = [c for c in X_raw.columns if c not in cat_cols and c not in bin_cols]

    preprocessor = preprocessing.get_preprocessor(cat_cols, num_cols, bin_cols)
    X_scaled = preprocessor.fit_transform(X_raw)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=4, random_state=config.RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save the complete clustering bundle
    clustering_bundle = {
        "preprocessor": preprocessor,
        "model": kmeans,
        "raw_features": list(X_raw.columns)
    }
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(clustering_bundle, os.path.join(config.MODEL_DIR, "best_clustering.joblib"), compress=3)
    print("Clustering bundle saved to models/best_clustering.joblib")
    
    # Compute metrics on a subsample of 5000 rows (for speed)
    sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
    X_sample = X_scaled[sample_idx]
    labels_sample = kmeans.labels_[sample_idx]
    
    sil = silhouette_score(X_sample, labels_sample)
    db = davies_bouldin_score(X_sample, labels_sample)
    ch = calinski_harabasz_score(X_sample, labels_sample)
    
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies Bouldin: {db:.4f}")
    print(f"Calinski Harabasz: {ch:.4f}")
    
    # Save clustering metrics
    df_metrics = pd.DataFrame({
        "Algorithm": ["KMeans"],
        "Silhouette": [sil],
        "Davies-Bouldin": [db],
        "Calinski-Harabasz": [ch]
    })
    df_metrics.to_csv(os.path.join(config.RESULTS_DIR, "clustering_metrics.csv"), index=False)

if __name__ == "__main__":
    train_clustering()
