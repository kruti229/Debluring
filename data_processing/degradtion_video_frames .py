import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def apply_augmentations(input_path, output_path, mode):
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] Processing folder: {input_path} with mode: {mode}")

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

        if mode == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

        elif mode == "compression":
            image.save(os.path.join(output_path, filename), format='JPEG', quality=random.randint(5, 25))
            count += 1
            continue

        elif mode == "brightness_contrast":
            brightness = ImageEnhance.Brightness(image)
            contrast = ImageEnhance.Contrast(image)
            image = brightness.enhance(random.uniform(0.3, 1.8))
            image = contrast.enhance(random.uniform(0.3, 1.8))

        image.save(os.path.join(output_path, filename))
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
                degradation_mode = random.choice(["blur", "compression", "brightness_contrast"])
                apply_augmentations(src_path, dst_path, degradation_mode)
    print(f"[COMPLETE] Input generation done.")

def compare_images(input_root: str, target_root: str):
    psnr_scores = []
    ssim_scores = []

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.endswith(".png"):
                continue

            input_img_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(input_img_path, input_root)
            target_img_path = os.path.join(target_root, rel_path)

            if not os.path.exists(target_img_path):
                continue

            input_img = cv2.imread(input_img_path)
            target_img = cv2.imread(target_img_path)

            if input_img is None or target_img is None or input_img.shape != target_img.shape:
                continue

            psnr_val = cv2.PSNR(target_img, input_img)
            ssim_val = ssim(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
                            data_range=255)

            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

            # Visualize one comparison
            input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            combined = np.hstack((target_rgb, input_rgb))
            plt.figure(figsize=(10, 5))
            plt.imshow(combined)
            plt.title(f"Target vs Input | PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
            plt.axis('off')
            plt.show()
            return  # show only one pair for visual check

    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")

# Uncomment to run
generate_input_from_target(
    "D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/target",
    "D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/testfolder"
)

# compare_images(
#     input_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/input_degraded_video",
#     target_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/target"
# )
