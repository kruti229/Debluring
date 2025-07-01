import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def degrade_video(video_path, output_path, effect_log):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mode = random.choice(["blur", "compression", "brightness_contrast", "network"])
    effect_log.append(f"{video_path} -> {output_path} : {mode}")
    print(f"[INFO] Applying {mode} to {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "blur":
            frame = cv2.GaussianBlur(frame, (5, 5), sigmaX=3)

        elif mode == "compression":
            _, encimg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(5, 25)])
            frame = cv2.imdecode(encimg, 1)

        elif mode == "brightness_contrast":
            alpha = random.uniform(0.3, 1.8)
            beta = random.randint(-60, 60)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        elif mode == "network":
            for _ in range(random.randint(5, 20)):
                x = random.randint(0, width - 20)
                y = random.randint(0, height - 20)
                w = random.randint(10, 50)
                h = random.randint(10, 50)
                frame[y:y+h, x:x+w] = random.randint(0, 255)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[DONE] Saved to {output_path}")

def process_all_videos(target_root, input_root, log_file):
    effect_log = []
    for subfolder in ["th", "th-bb", "th-m", "th-ob"]:
        target_subdir = os.path.join(target_root, subfolder)
        input_subdir = os.path.join(input_root, subfolder)
        if not os.path.exists(target_subdir):
            print(f"[SKIP] Missing: {target_subdir}")
            continue

        for filename in os.listdir(target_subdir):
            if not filename.endswith(".mp4"):
                continue

            input_video = os.path.join(target_subdir, filename)
            output_video = os.path.join(input_subdir, filename)
            degrade_video(input_video, output_video, effect_log)

    with open(log_file, "w") as f:
        for line in effect_log:
            f.write(line + "/n")
    print(f"[LOG] Saved degradation info to {log_file}")

# Example usage
process_all_videos(
    target_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/Videos/mp4/target_mp4",
    input_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/Videos/mp4/input_mp4",
    log_file="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/Videos/mp4/target_mp4/degradation_log.txt"
)
