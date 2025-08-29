import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
# static_image_mode=False: This indicates that it is a video stream rather than a static image, thereby improving processing efficiency.
# model_complexity=1: Model complexity, ranging from 0 to 2. The higher the number, the more accurate the detection but the slower the speed.

video_path = "D:/ERP/videos/chute01/cam2.avi"
cap = cv2.VideoCapture(video_path)

save_output = False

if save_output:
    out = cv2.VideoWriter("skeleton_output.avi",
                          cv2.VideoWriter_fourcc(*'XVID'),
                          int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Skeleton Drawing
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)。
        )

    cv2.imshow('Skeleton Detection', frame)
    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
pose.close()
