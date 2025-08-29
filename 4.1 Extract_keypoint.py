import pandas as pd
import os

csv_path = "D:/ERP/data_tuple3.csv"
frame_root = "D:/ERP/frames"

df = pd.read_csv(csv_path)

frame_info = []

for _, row in df.iterrows():
    chute = f"chute{int(row['chute']):02d}"
    cam = f"cam_{int(row['cam'])}"
    start = int(row['start'])
    end = int(row['end'])
    label = int(row['label'])

    for frame_id in range(start, end + 1):
        filename = f"{frame_id:04d}.jpg"
        full_path = os.path.join(frame_root, chute, cam, filename)
        frame_info.append((full_path, label))

print(f"load frames: {len(frame_info)}")


import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm
import os


mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True)

# Feature extraction function
def extract_keypoints(landmarks):
    if landmarks is None:
        return np.zeros(66)
    keypoints = []
    for lm in landmarks:
        keypoints.extend([lm.x, lm.y])
    return np.array(keypoints)

# Start processing each frame
features = []
labels = []

for path, label in tqdm(frame_info, desc="Extracting posture features"):
    if not os.path.exists(path):
        continue  # Some frames may be lost
    img = cv2.imread(path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)

    if results.pose_landmarks:
        vec = extract_keypoints(results.pose_landmarks.landmark)
    else:
        vec = np.zeros(66)  # Padding of empty frames (to ensure consistent sample count)

    features.append(vec)
    labels.append(label)

features_np = np.array(features)
labels_np = np.array(labels)

print(f"✅ feature shape: {features_np.shape}")
print(f"✅ feature shape: {labels_np.shape}")

np.save(r"D:/ERP/Project/ERP/pose_features.npy", features_np)
np.save(r"D:/ERP/Project/ERP/pose_labels.npy", labels_np)
np.save(r"D:/ERP/Project/ERP/frame_paths.npy", np.array([p for p, _ in frame_info])) # Save the original path of each frame

print("✅ Saved at D:/ERP/pose_features.npy and pose_labels.npy and frame_paths.npy")

