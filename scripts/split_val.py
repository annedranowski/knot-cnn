#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path("dataset")      # root dataset folder
TRAIN_DIR = BASE_DIR / "train"  # existing training set
VAL_DIR = BASE_DIR / "val"      # new validation folder
VAL_FRACTION = 0.1              # fraction of train moved to val
DRY_RUN = False                 # set True to preview without moving
# ----------------------------------------

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"❌ Train folder not found at {TRAIN_DIR}")
    
    # Create val directory if missing
    VAL_DIR.mkdir(exist_ok=True)

    print(f"🔄 Splitting validation set: {VAL_FRACTION*100:.0f}% from train/")
    print(f"Train path: {TRAIN_DIR}")
    print(f"Val path:   {VAL_DIR}\n")

    # Iterate over each class in train/
    for class_folder in TRAIN_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        print(f"📂 Processing class: {class_name}")

        # Create matching class folder in val/
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)

        # Get all images in this class
        images = list(class_folder.glob("*"))
        if not images:
            print(f"  ⚠️ No images found in {class_name}, skipping.")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_FRACTION))
        val_images = images[:n_val]

        print(f"  Found {len(images)} images → moving {n_val} to val/{class_name}")

        if not DRY_RUN:
            for img in val_images:
                shutil.move(str(img), str(VAL_DIR / class_name / img.name))

    print("\n✅ Done! Now you have:")
    for split in ["train", "val", "test"]:
        split_path = BASE_DIR / split
        if split_path.exists():
            total = sum(len(files) for _, _, files in os.walk(split_path))
            print(f"  {split:<5} → {total} images")

if __name__ == "__main__":
    main()
