# Perform clustering on the keypoint feature of the posture (KMeans), automatically select K, evaluate (Silhouette/NMI/ARI/Purity),
# and visualize (PCA two-dimensional) the confusion matrix and representative samples aligned with the output clustered-label pairs.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
import pandas as pd


FEATURES_NPY = r"D:/ERP/Project/ERP/pose_features.npy"
LABELS_NPY   = r"D:/ERP/Project/ERP/pose_labels.npy"
FRAMEPATHS_NPY = r"D:/ERP/Project/ERP/frame_paths.npy"

OUT_DIR = r"D:/ERP/Project/ERP/cluster_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Reading data
X_raw = np.load(FEATURES_NPY)  # (N, 66)
y_true = np.load(LABELS_NPY)   # (N,)

print("Load feature shape:", X_raw.shape)
print("Load label shape:", y_true.shape)

# Cleaning: Remove all-zero frames (Mediapipe failed frames) and record the index mapping
nonzero_mask = ~(np.all(X_raw == 0, axis=1))
X_raw = X_raw[nonzero_mask]
y_true = y_true[nonzero_mask]
kept_idx = np.where(nonzero_mask)[0]
print(f"delete No-instance samples {np.sum(~nonzero_mask)} , Saved {X_raw.shape[0]} .")

frame_paths = None
if os.path.exists(FRAMEPATHS_NPY):
    try:
        all_paths = np.load(FRAMEPATHS_NPY, allow_pickle=True)
        frame_paths = all_paths[nonzero_mask]
        print("Loading frame path succeeded:", len(frame_paths))
    except Exception as e:
        print("Loading the frame path failed. The sample export will be skipped.", e)

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)


# Select the Optimal K
K_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]
best_k, best_sil = None, -1
sil_scores = {}

for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels_k = km.fit_predict(X)
    sil = silhouette_score(X, labels_k)
    sil_scores[k] = sil
    if sil > best_sil:
        best_sil, best_k = sil, k

print("\nCandidate K's contour coefficient:")
for k in K_CANDIDATES:
    print(f"  K={k}: Silhouette={sil_scores[k]:.4f}")
print(f" Choose the best K={best_k}（Silhouette={best_sil:.4f}）")


# Re-cluster using the optimal K
kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
y_cluster = kmeans.fit_predict(X)

# Evaluation (Unsupervised Metrics + Consistency with Labels)
nmi = normalized_mutual_info_score(y_true, y_cluster)
ari = adjusted_rand_score(y_true, y_cluster)

# Calculate purity (majority voting method)
def purity_score(y_true, y_pred):
    contingency = pd.crosstab(y_true, y_pred)  # Row: Reality; Column: Clustering
    return np.sum(np.max(contingency.values, axis=0)) / np.sum(contingency.values)

purity = purity_score(y_true, y_cluster)

print("\nEvaluation index:")
print(f"  Silhouette: {best_sil:.4f}")
print(f"  NMI:        {nmi:.4f}")
print(f"  ARI:        {ari:.4f}")
print(f"  Purity:     {purity:.4f}")

# Optimal Matching: Map cluster IDs to label IDs (0/1)
# Use the Hungarian algorithm to maximize the number of correct matches (minimize the negative cross-table)
conf = pd.crosstab(y_true, y_cluster)  # shape: (n_labels, n_clusters)
cost = -conf.values  # Negation becomes the "cost" matrix
row_ind, col_ind = linear_sum_assignment(cost)
mapping = {}  # cluster_id -> mapped_label
for r, c in zip(row_ind, col_ind):
    mapping[c] = conf.index[r]

# Generate "Aligned Predicted Labels"
y_pred_aligned = np.array([mapping[c] for c in y_cluster])

# Aligned Confusion Matrix
conf_aligned = pd.crosstab(y_true, y_pred_aligned, rownames=['True'], colnames=['Pred'])
print("\nAligned confusion matrix:\n", conf_aligned) # 对齐后的混淆矩阵

# Save Confusion Matrix and Evaluation
conf_aligned.to_csv(os.path.join(OUT_DIR, "confusion_aligned.csv"), index=True)
with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"K={best_k}\n")
    f.write(f"Silhouette: {best_sil:.4f}\n")
    f.write(f"NMI:        {nmi:.4f}\n")
    f.write(f"ARI:        {ari:.4f}\n")
    f.write(f"Purity:     {purity:.4f}\n")


# Visualization: PCA to 2D (Cluster Coloring & True Label Coloring)
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X)

# 1) Color by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y_cluster, s=10, alpha=0.7, cmap="tab10")
plt.title(f"PCA Two-dimensional visualization (colored according to clusters), K={best_k}）") # 按照聚类上色
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_clusters.png"), dpi=300)
plt.show()

# 2) Color according to the actual labels
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y_true, s=10, alpha=0.7, cmap="coolwarm")
plt.title("PCA Two-dimensional visualization (colored according to the actual labels)") # 按真实颜色上色
plt.xlabel("PC1")
plt.ylabel("PC2")
cbar = plt.colorbar(scatter2, ticks=[0, 1])
cbar.ax.set_yticklabels(['No fall(0)', 'Fall(1)'])
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_labels.png"), dpi=300)
plt.show()


# Statistics of Clustering Clusters and Export of Representative Samples
summary_rows = []
for c in range(best_k):
    idx = np.where(y_cluster == c)[0]
    count = len(idx)
    frac = count / len(y_cluster)
    # Distribution of the true labels in each cluster
    dist = pd.Series(y_true[idx]).value_counts(normalize=True).to_dict()
    row = {"cluster": c, "count": count, "fraction": frac}
    row.update({f"true_{k}": v for k, v in dist.items()})
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows).sort_values("cluster")
summary_df.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"), index=False)
print("\nCluster summary (proportion of true labels in each cluster):\n", summary_df) # Clustering Summary (Proportion of True Labels per Cluster)

# Representative Sample
if frame_paths is not None:
    reps = []
    for c in range(best_k):
        idx = np.where(y_cluster == c)[0]
        if len(idx) == 0:
            continue
        # Select the sample in this cluster that is closest to its cluster center as the representative.
        center = kmeans.cluster_centers_[c]
        dists = np.linalg.norm(X[idx] - center, axis=1)
        rep_local = np.argmin(dists)
        rep_global_idx = idx[rep_local]
        reps.append({
            "cluster": c,
            "global_index": int(kept_idx[rep_global_idx]),  # Corresponding to the original data index
            "frame_path": str(frame_paths[rep_global_idx]),
            "true_label": int(y_true[rep_global_idx]),
            "aligned_pred": int(y_pred_aligned[rep_global_idx]),
        })
    reps_df = pd.DataFrame(reps).sort_values("cluster")
    reps_df.to_csv(os.path.join(OUT_DIR, "cluster_representatives.csv"), index=False)
    print("\n The representative samples of each cluster have been exported:cluster_representatives.csv") # Exported representative samples for each cluster
else:
    print("\n Not offer frame_paths.npy, skip.")
