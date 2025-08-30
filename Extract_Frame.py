import os
import cv2

def extract_frames(video_path, output_dir, fps_interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video :", video_path)
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps_interval == 0:
            out_path = os.path.join(output_dir, f"{saved_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[Finished] {video_path} → Number of frames to save: {saved_idx}")

def extract_all_chutes(video_root, output_root, fps_interval=1):
    for chute_folder in os.listdir(video_root):
        chute_path = os.path.join(video_root, chute_folder)
        if not os.path.isdir(chute_path):
            continue

        for file in os.listdir(chute_path):
            if file.endswith(".avi") and file.startswith("cam"):
                cam_id = file.replace("cam", "").replace(".avi", "")
                video_path = os.path.join(chute_path, file)

                output_dir = os.path.join(
                    output_root,
                    chute_folder.replace("chute", "chute_"),  # e.g. chute01 → chute_01
                    f"cam_{cam_id}"
                )

                extract_frames(video_path, output_dir, fps_interval)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch extract frames from videos")
    parser.add_argument("--video_root", required=True, help="Input video root directory")
    parser.add_argument("--output_root", required=True, help="Output frame root directory")
    parser.add_argument("--fps_interval", type=int, default=1, help="Save an image every how many frames (default: 1)")

    args = parser.parse_args()

    extract_all_chutes(args.video_root, args.output_root, args.fps_interval)

