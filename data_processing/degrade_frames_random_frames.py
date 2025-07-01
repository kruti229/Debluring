import os
import random
from PIL import Image, ImageEnhance, ImageFilter

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

# Update these paths as per your local directory
generate_input_from_target(
    "D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/target",
    "D:/Create/freelance/deblurring/data/VCD/video_data_VCD/frames/input"
)
