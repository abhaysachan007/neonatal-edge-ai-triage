"""
Build Train/Val/Test Manifest from OpenPOCUS folder structure.

Expected input folder layout:
  data/raw/
      normal/
          img001.png
          img002.png
          ...
      abnormal/
          img101.png
          ...

Output CSV (data/processed/phase1_manifest.csv):
  frame_path, label, patient_id, split

Patient-level split: 70% train / 15% val / 15% test
(all frames from one patient stay in the same split)

Usage:
  python scripts/build_manifest.py --input data/raw --output data/processed/phase1_manifest.csv
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


LABEL_MAP = {
    "normal": 0,
    "abnormal": 1,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def path_to_patient_id(path: Path) -> str:
    """
    Derive a pseudo patient_id from filename.
    If filenames follow pattern like PAT001_frame01.png, extracts PAT001.
    Otherwise, hashes the stem.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0]
    # fallback: hash-based pseudo ID
    return hashlib.md5(stem.encode()).hexdigest()[:8]


def assign_splits(patient_ids, train_frac=0.70, val_frac=0.15, seed=42):
    """Patient-level split assignment."""
    rng = np.random.default_rng(seed)
    unique_patients = sorted(set(patient_ids))
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    split_map = {}
    for i, pid in enumerate(unique_patients):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "validation"
        else:
            split_map[pid] = "test"

    return split_map


def build_manifest(input_dir: str, output_path: str, seed: int = 42):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_name, label in LABEL_MAP.items():
        class_dir = input_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Directory not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            patient_id = path_to_patient_id(img_path)
            rows.append({
                "frame_path": str(img_path.resolve()),
                "label": label,
                "label_name": class_name,
                "patient_id": patient_id,
            })

    if not rows:
        print("[ERROR] No images found. Check --input directory.")
        return

    df = pd.DataFrame(rows)
    split_map = assign_splits(df["patient_id"].tolist(), seed=seed)
    df["split"] = df["patient_id"].map(split_map)

    df.to_csv(output_path, index=False)

    print(f"\nManifest saved: {output_path}")
    print(f"Total frames: {len(df):,}")
    print("\nSplit distribution:")
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))

    # Leakage check
    for split_a, split_b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a = set(df[df["split"] == split_a]["patient_id"])
        b = set(df[df["split"] == split_b]["patient_id"])
        overlap = a & b
        if overlap:
            print(f"\n[ERROR] Patient leakage between {split_a} and {split_b}: {overlap}")
        else:
            print(f"[OK] No patient overlap: {split_a} vs {split_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="Folder with normal/ and abnormal/ subfolders")
    parser.add_argument("--output", default="data/processed/phase1_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifest(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
