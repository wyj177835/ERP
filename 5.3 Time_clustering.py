# Perform standardization, PCA visualization,
# KMeans clustering and evaluation (including Hungarian alignment, confusion matrix, and metrics) on the generated time_features
# Output:
#   {OUT_DIR}/time_pca2_labels.png
#   {OUT_DIR}/time_pca2_clusters.png
#   {OUT_DIR}/time_metrics.txt
#   {OUT_DIR}/time_confusion_aligned.csv
#   {OUT_DIR}/time_cluster_summary.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


IN_DIR  = r"D:/ERP/Project/ERP"
OUT_DIR = r"D:/ERP/Project/ERP/OUT"
os.makedirs(OUT_DIR, exist_ok=True)

# Which type of window label should be used as the comparison indicator: "majority" or "any"?
LABEL_KIND = "majority"   # or "any"

# Candidate K (used for selecting the optimal K)
K_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]

DRAW_ELBOW = True

X = np.load(os.path.join(IN_DIR, "time_features.npy"))             # (M, F)
y_major = np.load(os.path.join(IN_DIR, "time_labels_majority.npy"))# (M,)
y_any   = np.load(os.path.join(IN_DIR, "time_labels_any.npy"))     # (M,)
header  = pd.read_csv(os.path.join(IN_DIR, "time_features_header.csv"))["feature_name"].tolist()

if LABEL_KIND == "any":
    y_true = y_any
else:
    y_true = y_major

print(f"Loaded: X={X.shape}, y_true={y_true.shape} (LABEL_KIND={LABEL_KIND})")

if np.isnan(X).any():
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    print("NaN found and replaced with column means.")

# Standardization
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Choose K
best_k, best_sil = None, -1
sil_scores = {}

for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels_k = km.fit_predict(Xz)
    sil = silhouette_score(Xz, labels_k)
    sil_scores[k] = sil
    if sil > best_sil:
        best_sil, best_k = sil, k

print("\nCandidate K's silhouette:")
for k in K_CANDIDATES:
    print(f"  K={k}: Silhouette={sil_scores[k]:.4f}")
print(f"Choose the best K={best_k} (Silhouette={best_sil:.4f})")

if DRAW_ELBOW:
    sses = []
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xz)
        sses.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(K_CANDIDATES, sses, marker="o")
    plt.xlabel("K")
    plt.ylabel("SSE (inertia)")
    plt.title("Elbow (SSE vs K)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    elbow_png = os.path.join(OUT_DIR, "time_elbow_sse.png")
    plt.savefig(elbow_png, dpi=300)
    print(f"Elbow curve saved: {elbow_png}")
    plt.show()

# Use the optimal K clustering
kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
y_cluster = kmeans.fit_predict(Xz)

# PCA 2D Visualization
pca2 = PCA(n_components=2, random_state=42)
X2 = pca2.fit_transform(Xz)
print("\nPCA(2D) explained variance ratio:", pca2.explained_variance_ratio_,
      " | cum={:.2f}%".format(100*np.sum(pca2.explained_variance_ratio_)))

# Color according to the actual labels
plt.figure(figsize=(7,6))
sc1 = plt.scatter(X2[:,0], X2[:,1], c=y_true, s=10, alpha=0.7, cmap="coolwarm")
plt.title(f"PCA 2D (colored by TRUE label: {LABEL_KIND})")
plt.xlabel("PC1"); plt.ylabel("PC2")
cbar = plt.colorbar(sc1, ticks=[0,1])
cbar.ax.set_yticklabels(['0','1'])
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
pca_true_png = os.path.join(OUT_DIR, "time_pca2_labels.png")
plt.savefig(pca_true_png, dpi=300)
print(f"Saved: {pca_true_png}")
plt.show()

# Color by cluster ID
plt.figure(figsize=(7,6))
sc2 = plt.scatter(X2[:,0], X2[:,1], c=y_cluster, s=10, alpha=0.7, cmap="tab10")
plt.title(f"PCA 2D (colored by CLUSTER, K={best_k})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(sc2, label="Cluster ID")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
pca_cluster_png = os.path.join(OUT_DIR, "time_pca2_clusters.png")
plt.savefig(pca_cluster_png, dpi=300)
print(f"Saved: {pca_cluster_png}")
plt.show()

# Evaluation (NMI/ARI/Purity)
def purity_score(y_true_, y_pred_):
    tab = pd.crosstab(y_true_, y_pred_)
    return np.sum(np.max(tab.values, axis=0)) / np.sum(tab.values)

nmi = normalized_mutual_info_score(y_true, y_cluster)
ari = adjusted_rand_score(y_true, y_cluster)
purity = purity_score(y_true, y_cluster)

print("\nEvaluation:")
print(f"  Silhouette: {best_sil:.4f}")
print(f"  NMI:        {nmi:.4f}")
print(f"  ARI:        {ari:.4f}")
print(f"  Purity:     {purity:.4f}")

# Hungarian alignment & confusion matrix
conf = pd.crosstab(y_true, y_cluster)      # Row: True Label; Column: Cluster
cost = -conf.values
rind, cind = linear_sum_assignment(cost)    # Optimal Match
mapping = {c: conf.index[r] for r, c in zip(rind, cind)}  # cluster -> mapped label

y_pred_aligned = np.array([mapping[c] for c in y_cluster])
conf_aligned = pd.crosstab(y_true, y_pred_aligned, rownames=["True"], colnames=["Pred"])
print("\nAligned confusion matrix:\n", conf_aligned)

# Save
conf_aligned.to_csv(os.path.join(OUT_DIR, "time_confusion_aligned.csv"), index=True)
summary_rows = []
for c in range(best_k):
    idx = np.where(y_cluster == c)[0]
    dist = pd.Series(y_true[idx]).value_counts(normalize=True).to_dict()
    summary_rows.append({"cluster": c, "count": len(idx), **{f"true_{k}": v for k, v in dist.items()}})
pd.DataFrame(summary_rows).sort_values("cluster").to_csv(
    os.path.join(OUT_DIR, "time_cluster_summary.csv"), index=False
)

with open(os.path.join(OUT_DIR, "time_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"LABEL_KIND: {LABEL_KIND}\n")
    f.write(f"Best K: {best_k} | Silhouette: {best_sil:.4f}\n")
    f.write(f"NMI: {nmi:.4f}\nARI: {ari:.4f}\nPurity: {purity:.4f}\n")

print(f"\n✅ Done. Results saved in: {OUT_DIR}")
