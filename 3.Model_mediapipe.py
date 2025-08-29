import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models.video import r3d_18
import mediapipe as mp
import matplotlib.pyplot as plt

# Model Structure (3DCNN)
class Pure3DCNN(torch.nn.Module):
    def __init__(self):
        super(Pure3DCNN, self).__init__()
        self.backbone = r3d_18(pretrained=False)
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.backbone.fc.in_features, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))

# Calculation of bilateral framework angles
def calculate_average_torso_angle(landmarks):
    try:
        # Extract the two-dimensional coordinates of the left shoulder, left hip, and left knee.
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        # Extract the two-dimensional coordinates of the right shoulder, right hip, and right knee.
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])

        # Define a function for calculating the angle between two vectors
        def angle_between(v1, v2):
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))   # Calculate Cosine Similarity
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))  # Calculate Angle from Inverse Cosine
            return np.degrees(angle)  # Conversion from Radians to Degrees

        # Calculate the trunk angles of the left and right bodies separately
        left_angle = angle_between(left_hip - left_shoulder, left_knee - left_hip)
        right_angle = angle_between(right_hip - right_shoulder, right_knee - right_hip)

        return (left_angle + right_angle) / 2.0
    except:
        return 0


# State Fusion Logic
falling_counter = 0  # Counter for the number of frames where a fall occurred
cnn_counter = 0  # Counter for the number of frames predicted as falls by the CNN
safe_override_counter = 0  # Counter for the number of times the safety override was activated
FALLEN_THRESHOLD = 15  # Threshold for the number of frames required to reach the fall state
SAFE_RESET_TOLERANCE = 3  # Tolerance for transitioning from the FALLING state to the SAFE state

fallen_triggered = False

def update_state_machine(prev_state, angle, cnn_pred):
    global falling_counter, cnn_counter, safe_override_counter, fallen_triggered

    if prev_state == "SAFE":
        if 100 < angle < 140 and cnn_pred == 1:
            falling_counter += 1
            cnn_counter += 1
            if falling_counter >= FALLEN_THRESHOLD and cnn_counter >= FALLEN_THRESHOLD:
                fallen_triggered = True
                return "FALLEN"
            return "FALLING"
        else:
            falling_counter = 0
            cnn_counter = 0
            return "SAFE"

    elif prev_state == "FALLING":
        if cnn_pred == 1:
            falling_counter += 1
            cnn_counter += 1
            if falling_counter >= FALLEN_THRESHOLD and cnn_counter >= FALLEN_THRESHOLD:
                fallen_triggered = True
                return "FALLEN"
            return "FALLING"
        else:
            safe_override_counter += 1
            if safe_override_counter >= SAFE_RESET_TOLERANCE:
                falling_counter = 0
                cnn_counter = 0
                safe_override_counter = 0
                return "SAFE"
            return "FALLING"

    elif prev_state == "FALLEN":
        if cnn_pred == 0:
            return "SAFE"
        return "FALLEN"

    return "SAFE"

def prompt_user_args():
    import os

    while True:
        frame_dir = input('Please enter the frame directory path: ').strip()
        if os.path.isdir(frame_dir):
            break
        else:
            print("The directory does not exist. Please re-enter.")

    while True:
        s = input('Please enter the starting frame number: ').strip()
        if s.isdigit():
            start_id = int(s)
            break
        else:
            print("Please enter a valid integer.")

    while True:
        s = input('Please enter the end frame number: ').strip()
        if s.isdigit():
            end_id = int(s)
            break
        else:
            print("Please enter a valid integer.")

    if start_id > end_id:
        print(f"⚠️ Start Frame({start_id}) > End Frame({end_id}), It has been automatically swapped.")
        start_id, end_id = end_id, start_id

    return frame_dir, start_id, end_id

# Main
if __name__ == '__main__':
    try:
        frame_dir, start_id, end_id = prompt_user_args()

        model_path = "D:/ERP/Project/pythonProject/pure3dcnn_dropout.pth"

        import os, re
        frame_paths = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.lower().endswith(".jpg")
               and re.findall(r"(\d+)", f)
               and start_id <= int(re.findall(r"(\d+)", f)[-1]) <= end_id
        ], key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))

        if not frame_paths:
            raise RuntimeError(f"Between {start_id} and {end_id} can't find .jpg frame.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Pure3DCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        transform = Compose([Resize((112, 112)), ToTensor()])
        clip_window, torso_angle_series = [], []
        status = "SAFE"

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(static_image_mode=True)

        for idx, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # CNN
            pil_img = Image.fromarray(img_rgb)
            tensor_img = transform(pil_img)
            clip_window.append(tensor_img)
            if len(clip_window) == 30:
                input_tensor = torch.stack(clip_window).permute(1, 0, 2, 3).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = model(input_tensor).item()
                    cnn_pred = 1 if prob >= 0.5 else 0
                clip_window.pop(0)
            else:
                cnn_pred = 0

            # MediaPipe
            results = pose.process(img_rgb)
            angle = calculate_average_torso_angle(results.pose_landmarks.landmark) if results.pose_landmarks else 0
            torso_angle_series.append(angle)

            # State machine
            status = update_state_machine(status, angle, cnn_pred)

            # Visualization
            disp_img = cv2.resize(img, (640, 480))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(disp_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(disp_img, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255) if status != "SAFE" else (0, 255, 0), 3)

            if fallen_triggered:
                cv2.putText(disp_img, "A fall has occurred!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Fall Detection (Fusion)", disp_img)
            key = cv2.waitKey(40)
            if key == 27:
                break

        cv2.destroyAllWindows()
        pose.close()

        # Curved Angle
        plt.figure(figsize=(10, 4))
        plt.plot(torso_angle_series, label='Avg Torso Angle')
        plt.axhline(y=150, linestyle='--', label='Fallen Threshold')
        plt.axhline(y=140, linestyle='--', label='Falling Threshold')
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.title("Torso Angle Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Error: ", e)
