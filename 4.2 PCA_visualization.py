# Standardize the key point features of the posture, perform PCA dimensionality reduction, and visualize (color-coded according to labels)

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Data sources
USE_NPY = True

FEATURES_NPY = r"D:/ERP/Project/ERP/pose_features.npy"
LABELS_NPY   = r"D:/ERP/Project/ERP/pose_labels.npy"

if USE_NPY:
    features_np = np.load(FEATURES_NPY)  # Shape (N, 66)
    labels_np = np.load(LABELS_NPY)      # Shape (N,)
else:
    try:
        features_np  # noqa
        labels_np    # noqa
    except NameError:
        raise RuntimeError("cannot features_np / labels_np")

print("Primitive feature shape:", features_np.shape)
print("Primitive label shape:", labels_np.shape)

# Cleaning: Eliminate gestures that are completely zero (Mediapipe failure frames)
# It is said that pure zero vectors can affect standardization and PCA. One can choose to eliminate or retain them.
# I choose to eliminate them to facilitate the observation of the main structure.
mask_nonzero = ~(np.all(features_np == 0, axis=1))
X = features_np[mask_nonzero]
y = labels_np[mask_nonzero]
removed = np.sum(~mask_nonzero)
if removed > 0:
    print(f"Removed {removed} zero-instance samples(frames where posture recognition fails).Saved {X.shape[0]} Samples.")
else:
    print("No zero-instance samples were detected.")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: Start with 2-dimensional visualization first.
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)

print("\nPCA(2D) Explained Variance Ratio：", pca2.explained_variance_ratio_)
print("PC1 + PC2 Cumulative Explain Variance：{:.2f}%".format(100 * np.sum(pca2.explained_variance_ratio_)))

# The number of principal components required to achieve an 95% variance reduction
pca95 = PCA(n_components=0.95, random_state=42)
pca95.fit(X_scaled)
print("Number of principal components required to explain 95% of the variance：", pca95.n_components_)

# Visualization (colored according to labels)
plt.figure(figsize=(8, 6))
idx_fall = (y == 1)
idx_safe = (y == 0)

plt.scatter(X_pca2[idx_safe, 0], X_pca2[idx_safe, 1], s=12, alpha=0.6, label="No fall(0)")
plt.scatter(X_pca2[idx_fall, 0], X_pca2[idx_fall, 1], s=12, alpha=0.6, label="fall(1)")

plt.title("PCA Two-dimensional visualization (pose key points)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

out_png = r"D:/ERP/Project/ERP/pca_pose_scatter.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
print(f"Saved：{out_png}")
plt.show()

# Calculate the centroids of each class in the PCA space (for facilitating subsequent clustering comparison/identification)
def centroid(arr):
    return np.mean(arr, axis=0) if len(arr) > 0 else np.array([np.nan, np.nan])

c_safe = centroid(X_pca2[idx_safe])
c_fall = centroid(X_pca2[idx_fall])

print("\nPCA Spatial category centroid:")
print("No-fall(0) centroid:", c_safe)
print("fall(1)    centroid:", c_fall)
