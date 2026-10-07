import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import app.config as config
import app.preprocessing as preprocessing

def train_clustering():
    print("Training clustering model for Patient Segmentation (Review 2 sneak-peek/bonus)...")
    
    # Load and preprocess
    df = pd.read_csv(config.DATA_PATH)
    df_eng = preprocessing.engineer_features(df)
    
    # Select features for clustering (mix of raw and engineered features)
    cluster_features = [
        "age", "bmi", "income", "Hospital_Utilization_Score", 
        "Total_Chronic_Diseases", "Lifestyle_Risk_Score"
    ]
    
    X = df_eng[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=4, random_state=config.RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save the pipeline
    cluster_pipeline = {
        "scaler": scaler,
        "model": kmeans,
        "features": cluster_features
    }
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(cluster_pipeline, os.path.join(config.MODEL_DIR, "best_clustering.joblib"), compress=3)
    print("Clustering model saved to models/best_clustering.joblib")
    
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
