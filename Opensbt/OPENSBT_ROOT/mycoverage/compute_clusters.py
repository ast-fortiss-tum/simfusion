import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from mycoverage.compute_vector import compute_interaction_vector
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------
# Step 1: Load all simouts from folder
# -------------------------
def load_simouts_from_folder(folder_path):
    """
    Load all JSON simout files from a folder.
    Returns a list of loaded simout dicts.
    """
    simout_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    simout_files.sort()  # optional: ensure consistent order
    simout_list = []
    for path in simout_files:
        with open(path, "r") as f:
            simout_list.append(json.load(f))
    print(f"[INFO] Loaded {len(simout_list)} simout files from folder {folder_path}")
    return simout_list

# -------------------------
# Step 2: Compute Vectors
# -------------------------
def load_vectors_from_simouts(simout_list):
    vectors = []
    for idx, simout in enumerate(simout_list):
        print(f"[INFO] Computing vector for scenario {idx+1}/{len(simout_list)}")
        vec = compute_interaction_vector(simout)
        vectors.append(vec)
    return np.array(vectors)

# -------------------------
# Step 3: Standardization
# -------------------------
def standardize_vectors(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return (X - mean) / std

# -------------------------
# Step 4: Distance Matrix
# -------------------------
def compute_distance_matrix(X):
    dist_vector = pdist(X, metric='euclidean')
    dist_matrix = squareform(dist_vector)
    return dist_matrix

# -------------------------
# Step 5: DBSCAN Clustering
# -------------------------
def cluster_dbscan(X_std, eps=1.5, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    db.fit(X_std)
    return db.labels_

# -------------------------
# Full Pipeline
# -------------------------
def interaction_clustering_pipeline(folder_path, eps=1.5, min_samples=2):
    # Step 1: load simouts from folder
    simout_list = load_simouts_from_folder(folder_path)

    # Step 2: compute vectors
    X = load_vectors_from_simouts(simout_list)

    # Step 3: standardize
    X_std = standardize_vectors(X)

    # Step 4: distance matrix (optional)
    dist_matrix = compute_distance_matrix(X_std)
    print("[INFO] Pairwise distance matrix computed")

    # Step 5: DBSCAN clustering
    labels = cluster_dbscan(X_std, eps=eps, min_samples=min_samples)
    print("[INFO] DBSCAN clustering done")
    print(f"[INFO] Cluster labels: {labels}")

    return X_std, dist_matrix, labels
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Visualization function
# -------------------------
# -------------------------
# t-SNE visualization function
# -------------------------
def plot_clusters(X_std, labels, title="DBSCAN Clusters (t-SNE 2D)", perplexity=5, random_state=42, params=None):
    """
    Projects standardized vectors to 2D using t-SNE and plots them with cluster labels.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_2d = tsne.fit_transform(X_std)
    
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        cluster_points = X_2d[labels == lbl]
        if lbl == -1:
            # noise points
            plt.scatter(cluster_points[:,0], cluster_points[:,1], c='k', marker='x', label='noise')
        else:
            plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {lbl}')
    
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    suffix = f"_eps{params['eps']}_min_samples{params['min_samples']}" if params else ""
    plt.savefig(f"coverage/dbscan_clusters_p{perplexity}{suffix}.png")

    
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    folder = "/home/user/testing/topic1-opensbt-aw/results_wandb/Predict_GA_pop10_t06_00_00_k3_thc0.7_thg2.0_tho0.5_seed16/odlq83f0/simout_hifi"
    
    eps_values = [1.5]       # different neighborhood sizes
    min_samples_values = [2]          # different minimum points per cluster
    perplexity = 5                           # t-SNE perplexity
    
    simout_list = load_simouts_from_folder(folder)
    X = load_vectors_from_simouts(simout_list)
    X_std = standardize_vectors(X)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\n[INFO] DBSCAN eps={eps}, min_samples={min_samples}")
            labels = cluster_dbscan(X_std, eps=eps, min_samples=min_samples)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"[INFO] Number of clusters: {n_clusters}, Noise points: {n_noise}")
            print(f"[INFO] Cluster labels: {labels}")
            
            # Visualize clusters
            plot_clusters(
                X_std,
                labels,
                title=f"DBSCAN Clusters (t-SNE 2D) eps={eps}, min_samples={min_samples}",
                perplexity=perplexity,
                params={"eps": eps, "min_samples": min_samples}
            )