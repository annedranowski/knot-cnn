import os, random, shutil
from pathlib import Path

# Paths
base_dir = Path("dataset")
train_dir = base_dir / "train"
val_dir = base_dir / "val"

# Create val folder
val_dir.mkdir(exist_ok=True)

# Fraction for validation split
val_fraction = 0.1

# Iterate over each class (0, 3, 4, ...)
for class_folder in train_dir.iterdir():
    if not class_folder.is_dir():
        continue

    class_name = class_folder.name
    print(f"Processing class {class_name}...")

    # Make corresponding val class folder
    (val_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Get all images in this class
    images = list(class_folder.glob("*"))
    random.shuffle(images)

    # Take 10% for validation
    n_val = int(len(images) * val_fraction)
    val_images = images[:n_val]

    # Move val images
    for img in val_images:
        shutil.move(str(img), str(val_dir / class_name / img.name))

    print(f" → moved {n_val} images to val/{class_name}")

print("✅ Done! Now you have train/val/test splits.")