import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

def extract_frames(video_path, output_folder, fps=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_frame_count:03d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} into {output_folder}")

def process_videos_in_folder(root_folder, output_base):
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    target_subfolders = ["th", "th-bb", "th-m", "th-ob"]

    for subfolder in target_subfolders:
        current_path = os.path.join(root_folder, subfolder)
        for dirpath, _, filenames in os.walk(current_path):
            for file in filenames:
                if file.lower().endswith(video_extensions):
                    video_path = os.path.join(dirpath, file)
                    relative_path = os.path.relpath(dirpath, root_folder)
                    video_name = os.path.splitext(file)[0]
                    output_folder = os.path.join(output_base, relative_path, video_name)
                    extract_frames(video_path, output_folder)

def apply_augmentations(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] Processing folder: {input_path}")

    count = 0
    for filename in os.listdir(input_path):
        if not filename.endswith(".png"):
            continue

        input_file = os.path.join(input_path, filename)
        try:
            image = Image.open(input_file).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Cannot open {input_file}: {e}")
            continue

        # Stronger blur simulation
        if random.random() < 0.9:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

        # Heavier JPEG compression artifacts
        compressed_path = os.path.join(output_path, filename)
        if random.random() < 0.9:
            image.save(compressed_path, format='JPEG', quality=random.randint(5, 25))
        else:
            image.save(compressed_path)

        # More aggressive brightness/contrast changes
        if random.random() < 0.9:
            brightness = ImageEnhance.Brightness(image)
            contrast = ImageEnhance.Contrast(image)
            image = brightness.enhance(random.uniform(0.3, 1.8))
            image = contrast.enhance(random.uniform(0.3, 1.8))
            image.save(compressed_path)

        count += 1

    print(f"[DONE] Augmented {count} images → {output_path}")

def generate_input_from_target(target_root, input_root):
    print(f"[START] Generating input from: {target_root}")
    for subdir in ["th", "th-bb", "th-m", "th-ob"]:
        target_subdir = os.path.join(target_root, subdir)
        if not os.path.exists(target_subdir):
            print(f"[SKIP] Missing folder: {target_subdir}")
            continue
        for video_folder in os.listdir(target_subdir):
            src_path = os.path.join(target_subdir, video_folder)
            dst_path = os.path.join(input_root, subdir, video_folder)
            if os.path.isdir(src_path):
                apply_augmentations(src_path, dst_path)
    print(f"[COMPLETE] Input generation done.")
    
# Step 1: Extract raw frames from video dataset
process_videos_in_folder("D:/Create/freelance/deblurring/data/VCD/video_data/sharp/mp4", "D:/Create/freelance/deblurring/data/VCD/video_data/frames/target")

# Step 2: Generate degraded input frames from clean targets
# This simulates low-quality input for supervised learning (input -> target)
generate_input_from_target("D:/Create/freelance/deblurring/data/VCD/video_data/frames/target", "D:/Create/freelance/deblurring/data/VCD/video_data/frames/input")
