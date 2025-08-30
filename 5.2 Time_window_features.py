# Convert "single-frame angle/proportion features" into "time-window statistical features"
# Input:
#   D:/ERP/angle_features.npy          -> (N, D)
#   D:/ERP/angle_labels.npy            -> (N,)
#   D:/ERP/frame_paths.npy             -> (N,)
#   D:/ERP/angle_features_header.csv   -> Feature name
# output:
#   D:/ERP/time_features.npy           -> (M, F)  Window-level features
#   D:/ERP/time_labels_majority.npy    -> (M,)    Majority voting labels
#   D:/ERP/time_labels_any.npy         -> (M,)    As long as there is a "1" shown within the window, it will be recorded as 1.
#   D:/ERP/time_features_header.csv    -> Feature name (with statistical suffix)

import os
import numpy as np
import pandas as pd

IN_DIR  = r"D:/ERP/Project/ERP"
OUT_DIR = r"D:/ERP/Project/ERP"
os.makedirs(OUT_DIR, exist_ok=True)

ANGLE_FEATURES_NPY = os.path.join(IN_DIR, "angle_features.npy")
ANGLE_LABELS_NPY   = os.path.join(IN_DIR, "angle_labels.npy")
FRAME_PATHS_NPY    = os.path.join(IN_DIR, "frame_paths_Angle.npy")
HEADER_CSV         = os.path.join(IN_DIR, "angle_features_header.csv")

# Sliding Window Parameters (Can be adjusted as needed)
WINDOW = 30    # Window Length (Frames)
STRIDE = 5     # Step Length (Frame)
MAX_INTERP_GAP = 5  # Maximum consecutive missing length (frames) allowing interpolation

# Read Data
X = np.load(ANGLE_FEATURES_NPY)   # (N, D)
y = np.load(ANGLE_LABELS_NPY)     # (N,)
paths = np.load(FRAME_PATHS_NPY, allow_pickle=True)  # (N,)
header = pd.read_csv(HEADER_CSV)["feature_name"].tolist()  # LengthD

N, D = X.shape
print(f"Loaded angle features: X={X.shape}, y={y.shape}, paths={len(paths)}, D={len(header)}")

# Missing Value Handling (NaN values come from failed pose frames)
# Strategy: Perform "limited-length linear interpolation" for each dimension.
# The beginning and end are filled with the most recent values. If still empty, use the mean value of the entire column.
X_filled = X.copy()
for d in range(D):
    col = pd.Series(X_filled[:, d], dtype="float64")

    # Mark the lengths of consecutive missing segments. If the length exceeds the threshold, do not insert.
    is_na = col.isna().values
    if is_na.any():
        # Only perform linear interpolation for the missing segments that are less than or equal to MAX_INTERP_GAP
        # Method: First, perform global interpolation, then restore the long segments that exceed the threshold to NaN,
        # and finally fill in the gaps at the beginning and end
        col_interp = col.interpolate(method="linear", limit=MAX_INTERP_GAP, limit_direction="both")
        col = col_interp
        col = col.fillna(method="ffill").fillna(method="bfill")

    # If there are still NaN values, replace them with the mean value of this column (in extreme cases)
    if col.isna().any():
        col = col.fillna(col.mean())

    X_filled[:, d] = col.values.astype(np.float32)

# Statistical function
def window_stats(mat):  # mat: (T, D)
    """
    return (stats_features: 1×(D*5), names with suffix)
    Statistic:mean, std, min, max, range;And the trend slope and the maximum first-order difference
    """
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    mn   = np.nanmin(mat, axis=0)
    mx   = np.nanmax(mat, axis=0)
    rg   = mx - mn
    base_feats = np.stack([mean, std, mn, mx, rg], axis=0)  # (5, D)

    names = []
    for n in header:
        names += [f"{n}_mean", f"{n}_std", f"{n}_min", f"{n}_max", f"{n}_range"]

    # Linear trend (slope): The slope of a linear regression for each dimension based on the t-value.
    t = np.arange(mat.shape[0], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)  # Standardized timeline, avoiding the influence of scale
    slope = []
    for d in range(mat.shape[1]):
        yv = mat[:, d]
        # polyfit once: Returns [slope, intercept]
        sl = np.polyfit(t, yv, 1)[0] if np.isfinite(yv).all() else 0.0
        slope.append(sl)
    slope = np.array(slope, dtype=np.float32)[None, :]  # (1, D)
    base_feats = np.concatenate([base_feats, slope], axis=0)
    names += [f"{n}_slope" for n in header]

    # Maximum rate of change (max_abs_diff): Capturing abrupt changes
    diff = np.diff(mat, axis=0)  # (T-1, D)
    mad  = np.nanmax(np.abs(diff), axis=0)[None, :]
    base_feats = np.concatenate([base_feats, mad], axis=0)
    names += [f"{n}_max_abs_diff" for n in header]

    return base_feats.reshape(-1), names  # (D*7,), list

# Feature Enhancement: Add "Exclusive Time Feature" to Key Fields
# If these names exist, additional window features such as speed/maximum values will be derived.
SPECIAL_FIELDS = {
    "torso_tilt_deg": ["vel_mean", "vel_min", "vel_max"],
    "neck_tilt_deg":  ["vel_mean", "vel_min", "vel_max"],
    "center_y_minus_shoulder_mid": ["vel_mean", "vel_min", "vel_max"],  # Common Falling Speeds
}
name_to_idx = {n: i for i, n in enumerate(header)}

def extra_window_features(mat):
    feats = []
    names = []
    for fname, ops in SPECIAL_FIELDS.items():
        if fname not in name_to_idx:
            continue
        idx = name_to_idx[fname]
        series = mat[:, idx]
        vel = np.diff(series)  # First-order difference approximation of velocity
        if "vel_mean" in ops:
            feats.append(np.nanmean(vel))
            names.append(f"{fname}_vel_mean")
        if "vel_min" in ops:
            feats.append(np.nanmin(vel))
            names.append(f"{fname}_vel_min")
        if "vel_max" in ops:
            feats.append(np.nanmax(vel))
            names.append(f"{fname}_vel_max")
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.float32), []
    return np.array(feats, dtype=np.float32), names

win_feats = []
win_labels_major = []
win_labels_any = []
meta_rows = []

i = 0
while i + WINDOW <= N:
    seg = X_filled[i:i+WINDOW, :]   # (W, D)
    f1, base_names = window_stats(seg)
    f2, extra_names = extra_window_features(seg)
    feat = np.concatenate([f1, f2], axis=0)

    # Tag Strategy
    y_seg = y[i:i+WINDOW]
    # Majority Vote
    maj = int(np.round(y_seg.mean()))  # 0/1 Majority (Equal votes are rounded down automatically)
    # As long as there is a 1, it is considered as 1 (which is more conducive to the recall of "falling segment")
    any1 = int((y_seg > 0).any())

    win_feats.append(feat)
    win_labels_major.append(maj)
    win_labels_any.append(any1)

    meta_rows.append({
        "start_idx": i,
        "end_idx": i+WINDOW-1,
        "start_path": str(paths[i]),
        "end_path": str(paths[i+WINDOW-1]),
        "label_majority": maj,
        "label_any": any1
    })

    i += STRIDE

win_feats = np.vstack(win_feats).astype(np.float32)           # (M, F)
win_labels_major = np.array(win_labels_major, dtype=np.int64) # (M,)
win_labels_any   = np.array(win_labels_any, dtype=np.int64)   # (M,)

# Feature Name Concatenation
full_header = base_names + extra_names

print("\n=== Window features completed ===")
print("time_features:", win_feats.shape)
print("labels_majority:", win_labels_major.shape)
print("labels_any:", win_labels_any.shape)
print(f"Number of windows M = {(N - WINDOW)//STRIDE + 1}")

# Save
np.save(os.path.join(OUT_DIR, "time_features.npy"), win_feats)
np.save(os.path.join(OUT_DIR, "time_labels_majority.npy"), win_labels_major)
np.save(os.path.join(OUT_DIR, "time_labels_any.npy"), win_labels_any)
pd.Series(full_header, name="feature_name").to_csv(
    os.path.join(OUT_DIR, "time_features_header.csv"),
    index=False, encoding="utf-8"
)
pd.DataFrame(meta_rows).to_csv(
    os.path.join(OUT_DIR, "time_window_meta.csv"),
    index=False, encoding="utf-8"
)

print(f"\n✅ Saved at\n  {OUT_DIR}/time_features.npy"
      f"\n  {OUT_DIR}/time_labels_majority.npy"
      f"\n  {OUT_DIR}/time_labels_any.npy"
      f"\n  {OUT_DIR}/time_features_header.csv"
      f"\n  {OUT_DIR}/time_window_meta.csv")
