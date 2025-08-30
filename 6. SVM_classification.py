# Perform SVM classification on the "time_features" (supervised learning), and compare the results with the unsupervised ones.
# Output:
#   svm_metrics.txt
#   svm_confusion_matrix.png
#   svm_roc_curve.png

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay


IN_DIR  = r"D:/ERP/Project/ERP"
OUT_DIR = r"D:/ERP/Project/ERP/OUT"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_KIND = "majority"

X = np.load(os.path.join(IN_DIR, "time_features.npy"))
y_major = np.load(os.path.join(IN_DIR, "time_labels_majority.npy"))
y_any   = np.load(os.path.join(IN_DIR, "time_labels_any.npy"))

if LABEL_KIND == "any":
    y = y_any
else:
    y = y_major

print(f"Loaded: X={X.shape}, y={y.shape} (LABEL_KIND={LABEL_KIND})")

# Standardization
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Define the SVM classifier
clf = SVC(kernel="rbf", probability=True, random_state=42)

# Ten-fold cross-validation prediction
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, Xz, y, cv=cv)
y_proba = cross_val_predict(clf, Xz, y, cv=cv, method="predict_proba")[:,1]

# Index calculation
acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec  = recall_score(y, y_pred)
f1   = f1_score(y, y_pred)
auc  = roc_auc_score(y, y_proba)

print("\nSVM 10-fold CV Results:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-score:  {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")

# Save
with open(os.path.join(OUT_DIR, "svm_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"LABEL_KIND: {LABEL_KIND}\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall:    {rec:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {auc:.4f}\n")

# Draw the confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-fall","Fall"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"SVM Confusion Matrix ({LABEL_KIND})")
cm_path = os.path.join(OUT_DIR, "svm_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()
print(f"Confusion matrix saved: {cm_path}")

# ROC
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"SVM (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SVM)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
roc_path = os.path.join(OUT_DIR, "svm_roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.show()
print(f"ROC curve saved: {roc_path}")

print("\n✅ Done. Results saved in:", OUT_DIR)
