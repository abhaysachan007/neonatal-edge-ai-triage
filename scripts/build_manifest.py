"""
Build Train/Val/Test Manifest from OpenPOCUS folder structure.

Expected input folder layout:
  data/raw/
      normal/
          img001.png
          ...
      abnormal/
          img101.png
          ...

Output CSV (data/processed/phase1_manifest.csv):
  frame_path, label, label_name, patient_id, split

Patient-level STRATIFIED split: 70% train / 15% val / 15% test
Stratification: normal/abnormal ratio preserved in each split.

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
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return hashlib.md5(stem.encode()).hexdigest()[:8]


def assign_splits_stratified(df: pd.DataFrame, train_frac: float = 0.70,
                              val_frac: float = 0.15, seed: int = 42) -> dict:
    """
    Patient-level stratified split.
    For each class, split its patients 70/15/15 independently, then merge.
    Preserves normal/abnormal ratio in each split.
    """
    rng = np.random.default_rng(seed)

    patient_label = df.groupby("patient_id")["label"].agg(
        lambda x: int(x.mode()[0])
    )

    split_map: dict = {}
    for label_val in sorted(patient_label.unique()):
        patients = sorted(patient_label[patient_label == label_val].index.tolist())
        rng.shuffle(patients)
        n = len(patients)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        for i, pid in enumerate(patients):
            if i < n_train:
                split_map[pid] = "train"
            elif i < n_train + n_val:
                split_map[pid] = "validation"
            else:
                split_map[pid] = "test"

    return split_map


def build_manifest(input_dir: str, output_path: str, seed: int = 42):
    input_dir   = Path(input_dir)
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
            rows.append({
                "frame_path": str(img_path.resolve()),
                "label":      label,
                "label_name": class_name,
                "patient_id": path_to_patient_id(img_path),
            })

    if not rows:
        print("[ERROR] No images found. Check --input directory.")
        return

    df = pd.DataFrame(rows)
    split_map = assign_splits_stratified(df, train_frac=0.70, val_frac=0.15, seed=seed)
    df["split"] = df["patient_id"].map(split_map)

    df.to_csv(output_path, index=False)

    print(f"\nManifest saved: {output_path}")
    print(f"Total frames: {len(df):,}")
    print("\nSplit x label distribution:")
    dist = df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    print(dist)
    pct = dist.apply(lambda r: (r / r.sum() * 100).round(1), axis=1)
    print("\nClass % per split:")
    print(pct)

    for split_a, split_b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a = set(df[df["split"] == split_a]["patient_id"])
        b = set(df[df["split"] == split_b]["patient_id"])
        overlap = a & b
        status = f"[ERROR] LEAKAGE: {overlap}" if overlap else "[OK] No patient overlap"
        print(f"  {status}: {split_a} vs {split_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/processed/phase1_manifest.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    build_manifest(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
