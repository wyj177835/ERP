# For each frame, calculate the "angle + normalized geometry" single-frame feature and save it as .npy/.csv
# Output:
#   D:/ERP/angle_features.npy   -> (N, D) feature matrix
#   D:/ERP/angle_labels.npy     -> (N,)   labels (0/1)
#   D:/ERP/frame_paths.npy      -> (N,)   frame path (for convenient subsequent visualization)
#   D:/ERP/angle_features_header.csv -> feature names (for convenience in the paper table)

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

CSV_PATH    = r"D:/ERP/data_tuple3.csv"
FRAMES_ROOT = r"D:/ERP/frames"
OUT_DIR     = r"D:/ERP/Project/ERP"
os.makedirs(OUT_DIR, exist_ok=True)

# Read the CSV file and convert it into a list of frames
df = pd.read_csv(CSV_PATH)
frame_info = []  # [(full_path, label), ...]
for _, row in df.iterrows():
    chute = f"chute{int(row['chute']):02d}"
    cam   = f"cam_{int(row['cam'])}"
    start = int(row['start']); end = int(row['end'])
    label = int(row['label'])
    for fid in range(start, end + 1):
        filename = f"{fid:04d}.jpg"
        frame_info.append((os.path.join(FRAMES_ROOT, chute, cam, filename), label))

print(f"Total number of frames to be processed: {len(frame_info)}")

# Utility functions for angles/distances
mp_pose = mp.solutions.pose
POSE_LM = mp_pose.PoseLandmark

def get_xy(lms, idx):
    lm = lms[idx.value]
    return np.array([lm.x, lm.y], dtype=np.float32), float(lm.visibility)

def angle_at(a, b, c):
    """
    The angle (in degrees) between the return point b and the vertex. a, b, c are 2D coordinates (x, y).
    """
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def inclination_deg(v, ref="vertical"):
    """
    Calculate the inclination angle (in degrees) of vector v relative to the reference axis. ref: 'vertical' or 'horizontal'
    """
    if np.linalg.norm(v) < 1e-6:
        return np.nan
    vx, vy = v[0], v[1]
    # Image coordinates: x right, y bottom (Mediapipe uses normalized coordinates; y decreases as it goes downward)
    if ref == "vertical":
        # Relative angle to the vertical axis: 0 degrees = vertical, 90 degrees = horizontal
        ang = np.degrees(np.arctan2(abs(vx), abs(vy)))
    else:
        # Relative angle with the horizontal axis
        ang = np.degrees(np.arctan2(abs(vy), abs(vx)))
    return float(ang)

def safe_ratio(d_num, d_den):
    return float(d_num / d_den) if d_den > 1e-6 else np.nan

# Single-frame feature calculation
def compute_single_frame_features(landmarks):
    """
    Input: 33 2D key points of mediapipe
    Output: Feature vector (list) and feature names (list) of one frame
    """
    names = []
    vals  = []

    # Key Points (Common)
    ls, vis_ls = get_xy(landmarks, POSE_LM.LEFT_SHOULDER)
    rs, vis_rs = get_xy(landmarks, POSE_LM.RIGHT_SHOULDER)
    lh, vis_lh = get_xy(landmarks, POSE_LM.LEFT_HIP)
    rh, vis_rh = get_xy(landmarks, POSE_LM.RIGHT_HIP)
    le, vis_le = get_xy(landmarks, POSE_LM.LEFT_ELBOW)
    re, vis_re = get_xy(landmarks, POSE_LM.RIGHT_ELBOW)
    lw, vis_lw = get_xy(landmarks, POSE_LM.LEFT_WRIST)
    rw, vis_rw = get_xy(landmarks, POSE_LM.RIGHT_WRIST)
    lk, vis_lk = get_xy(landmarks, POSE_LM.LEFT_KNEE)
    rk, vis_rk = get_xy(landmarks, POSE_LM.RIGHT_KNEE)
    la, vis_la = get_xy(landmarks, POSE_LM.LEFT_ANKLE)
    ra, vis_ra = get_xy(landmarks, POSE_LM.RIGHT_ANKLE)
    nose, vis_ns= get_xy(landmarks, POSE_LM.NOSE)

    # Visibility Mean (Can Be Used as a Quality Indicator)
    mean_vis = np.mean([vis_ls,vis_rs,vis_lh,vis_rh,vis_le,vis_re,vis_lw,vis_rw,vis_lk,vis_rk,vis_la,vis_ra,vis_ns])

    # Midpoint and Dimensions (Standardized Proportions)
    s_mid = 0.5 * (ls + rs)   # Shoulder Center Point
    h_mid = 0.5 * (lh + rh)   # Hip Center
    torso_vec = h_mid - s_mid
    shoulder_w = np.linalg.norm(rs - ls)   # Shoulder Width
    hip_w      = np.linalg.norm(rh - lh)   # Hip Width
    torso_len  = np.linalg.norm(torso_vec) # Length of the torso

    # Angle-related features
    # 1) The relative vertical inclination angle of the trunk (the greater the value, the more "flat/laid-back" it is)
    names += ["torso_tilt_deg"]
    vals  += [inclination_deg(torso_vec, ref="vertical")]

    # 2) The inclination angle (head posture) of the neck direction (from the midpoint of the shoulders to the nose), which is relatively vertical
    names += ["neck_tilt_deg"]
    vals  += [inclination_deg(nose - s_mid, ref="vertical")]

    # 3) Hip-Knee-Ankle Angle (Degree of Knee Flexion on Both Sides)
    names += ["knee_angle_L_deg", "knee_angle_R_deg"]
    vals  += [angle_at(lh, lk, la), angle_at(rh, rk, ra)]

    # 4) Shoulder-Erector-Wrist Angle (Degree of Elbow Flexion on Both Sides)
    names += ["elbow_angle_L_deg", "elbow_angle_R_deg"]
    vals  += [angle_at(ls, le, lw), angle_at(rs, re, rw)]

    # 5) Shoulder-Hip-Knee Angle (Degree of Hip Flexion on Both Sides)
    names += ["hip_angle_L_deg", "hip_angle_R_deg"]
    vals  += [angle_at(ls, lh, lk), angle_at(rs, rh, rk)]

    # 6) The body is generally inclined at a relatively horizontal angle (from the midpoint of the shoulders to the midpoint of the hips is approximately horizontal)
    names += ["torso_inclination_to_horizontal_deg"]
    vals  += [inclination_deg(torso_vec, ref="horizontal")]

    # Standardized distance/proportion (eliminating the influence of scale/position)
    # 7) Foot spacing / Shoulder width, Hip width
    ankle_dist = np.linalg.norm(ra - la)
    names += ["ankle_dist_over_shoulder", "ankle_dist_over_hip"]
    vals  += [safe_ratio(ankle_dist, shoulder_w), safe_ratio(ankle_dist, hip_w)]

    # 8) Length of the torso / Shoulder width (Posture scale relationship)
    names += ["torso_over_shoulder"]
    vals  += [safe_ratio(torso_len, shoulder_w)]

    # 9) The approximate vertical position of the body's "center of gravity" (the greater the value of y, the lower it is), the difference relative to the midpoint of the nose/shoulder
    names += ["center_y_minus_nose", "center_y_minus_shoulder_mid"]
    vals  += [float(h_mid[1] - nose[1]), float(h_mid[1] - s_mid[1])]

    # 10) Mean of key point visibility (Quality control / Feature extraction is also possible)
    names += ["mean_visibility"]
    vals  += [float(mean_vis)]

    return names, vals

# Main processing loop
pose = mp_pose.Pose(static_image_mode=True)
feature_list = []
label_list   = []
path_list    = []

invalid_cnt = 0

pbar = tqdm(frame_info, desc="Computing angle features")
for fpath, label in pbar:
    if not os.path.exists(fpath):
        invalid_cnt += 1
        continue
    img = cv2.imread(fpath)
    if img is None:
        invalid_cnt += 1
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        # If recognition fails: Fill with NaN (Options for subsequent processing include elimination or interpolation)
        if len(feature_list) == 0:
            # Initialize the feature names once
            fnames, _ = compute_single_frame_features([type("Lm", (), {"x":0,"y":0,"visibility":0})]*33)
        _, vals = fnames, [np.nan]*len(fnames)
        feature_list.append(vals)
        label_list.append(label)
        path_list.append(fpath)
        continue

    fnames, vals = compute_single_frame_features(res.pose_landmarks.landmark)
    feature_list.append(vals)
    label_list.append(label)
    path_list.append(fpath)

pose.close()

features = np.array(feature_list, dtype=np.float32)  # (N, D)
labels   = np.array(label_list, dtype=np.int64)      # (N,)
paths    = np.array(path_list, dtype=object)         # (N,)

print("Feature matrix shape:", features.shape)
print("Label vector shape:", labels.shape)
print("Number of paths     :", len(paths))
print("Number of failed read frames :", invalid_cnt)

# Save output
np.save(os.path.join(OUT_DIR, "angle_features.npy"), features)
np.save(os.path.join(OUT_DIR, "angle_labels.npy"), labels)
np.save(os.path.join(OUT_DIR, "frame_paths_Angle.npy"), paths)

# Save feature name
pd.Series(fnames, name="feature_name").to_csv(
    os.path.join(OUT_DIR, "angle_features_header.csv"),
    index=False, encoding="utf-8"
)

print(f"\n✅ Saved：\n  {OUT_DIR}/angle_features.npy\n  {OUT_DIR}/angle_labels.npy\n  {OUT_DIR}/frame_paths_Angle.npy\n  {OUT_DIR}/angle_features_header.csv")

nan_rows = np.isnan(features).any(axis=1)
print(f"The number of samples containing NaN : {np.sum(nan_rows)} (When Mediapipe fails or the visibility is too low, this will occur.)")
