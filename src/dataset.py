"""
dataset.py
----------
Helper utilities for preparing and validating the dataset before training.

Run this script to:
  1. Check your folder structure is correct
  2. Verify label files match image files
  3. Print a class distribution summary
  4. Split a folder of images into train/val/test sets

Usage:
    python src/dataset.py --check          # Validate existing dataset
    python src/dataset.py --split DIR      # Split raw images into train/val/test
"""

import os
import argparse
import random
import shutil
from pathlib import Path
from collections import Counter

# ── Class names — keep in sync with data.yaml ──────────────────────────────────
CLASS_NAMES = {
    0: "worker",
    1: "hard-hat",
    2: "safety-vest",
    3: "no-hard-hat",
    4: "no-safety-vest",
    5: "safety-harness",
}

SPLITS = ["train", "val", "test"]


# ── Dataset validation ──────────────────────────────────────────────────────────

def check_dataset(data_dir: str = "data"):
    """
    Checks that every image has a corresponding label file and vice versa.
    Prints a class distribution table.
    """
    print("=" * 50)
    print("Dataset Validation Report")
    print("=" * 50)

    total_images = 0
    total_labels = 0
    class_counts = Counter()

    for split in SPLITS:
        img_dir = Path(data_dir) / "images" / split
        lbl_dir = Path(data_dir) / "labels" / split

        if not img_dir.exists():
            print(f"[WARN] Missing folder: {img_dir}")
            continue

        images = set(p.stem for p in img_dir.glob("*") if p.suffix in {".jpg", ".jpeg", ".png"})
        labels = set(p.stem for p in lbl_dir.glob("*.txt")) if lbl_dir.exists() else set()

        missing_labels = images - labels
        missing_images = labels - images

        print(f"\n[{split.upper()}]")
        print(f"  Images : {len(images)}")
        print(f"  Labels : {len(labels)}")

        if missing_labels:
            print(f"  [WARN] {len(missing_labels)} images have no label file:")
            for name in list(missing_labels)[:5]:
                print(f"    - {name}")

        if missing_images:
            print(f"  [WARN] {len(missing_images)} label files have no matching image:")
            for name in list(missing_images)[:5]:
                print(f"    - {name}")

        # Count class occurrences in labels
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1

        total_images += len(images)
        total_labels += len(labels)

    print(f"\n[TOTAL] Images: {total_images} | Labels: {total_labels}")
    print("\nClass Distribution:")
    print(f"  {'Class':<20} {'Count':>8}")
    print(f"  {'-'*30}")
    for class_id in sorted(class_counts):
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        count = class_counts[class_id]
        print(f"  {name:<20} {count:>8}")

    if not class_counts:
        print("  [WARN] No annotation data found. Have you added label files?")


# ── Train/val/test split ────────────────────────────────────────────────────────

def split_dataset(raw_dir: str, data_dir: str = "data",
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.1,
                  seed: int = 42):
    """
    Takes a folder of images (with matching .txt label files) and splits them
    into train / val / test folders inside data/.

    Expects raw_dir to contain:
        raw_dir/
          image1.jpg
          image1.txt      <-- YOLO label file with same name
          image2.jpg
          image2.txt
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    raw_path = Path(raw_dir)
    images = sorted([
        p for p in raw_path.glob("*")
        if p.suffix in {".jpg", ".jpeg", ".png"}
    ])

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    split_map = (
        [(img, "train") for img in images[:n_train]] +
        [(img, "val")   for img in images[n_train:n_train + n_val]] +
        [(img, "test")  for img in images[n_train + n_val:]]
    )

    print(f"\nSplitting {n} images: "
          f"{n_train} train / {n_val} val / {n - n_train - n_val} test")

    for img_path, split in split_map:
        lbl_path = img_path.with_suffix(".txt")

        dst_img = Path(data_dir) / "images" / split / img_path.name
        dst_lbl = Path(data_dir) / "labels" / split / lbl_path.name

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            print(f"  [WARN] No label for {img_path.name}, copying image only.")

    print("[DONE] Dataset split complete.")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument("--check", action="store_true",
                        help="Validate current dataset structure")
    parser.add_argument("--split", type=str, metavar="RAW_DIR",
                        help="Split raw images folder into train/val/test")
    args = parser.parse_args()

    if args.check:
        check_dataset()
    elif args.split:
        split_dataset(raw_dir=args.split)
    else:
        parser.print_help()
