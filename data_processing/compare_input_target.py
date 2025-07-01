import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
    

# Example usage
compare_images(
    input_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/input",
    target_root="D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/target"
)
