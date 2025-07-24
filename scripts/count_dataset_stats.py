#!/usr/bin/env python3
import os
from pathlib import Path
from collections import Counter

# Base dataset path
BASE_DIR = Path("dataset")  # adjust if needed
SPLITS = ["train", "test"]  # current splits

def count_images(folder: Path):
    counts = Counter()
    for cls in sorted(os.listdir(folder)):
        cls_path = folder / cls
        if cls_path.is_dir():
            n_imgs = len([f for f in os.listdir(cls_path) if (cls_path / f).is_file()])
            counts[cls] = n_imgs
    return counts

def main():
    for split in SPLITS:
        split_path = BASE_DIR / split
        if not split_path.exists():
            print(f"⚠️ Split '{split}' not found, skipping...")
            continue

        print(f"\n📂 {split.upper()} SPLIT")
        counts = count_images(split_path)
        total = sum(counts.values())
        
        for cls, n in counts.items():
            print(f"  Class {cls:>3}: {n} images")
        print(f"  ➡ TOTAL: {total} images\n")

if __name__ == "__main__":
    main()
