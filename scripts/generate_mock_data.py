"""
Generate mock LUS images for pipeline validation.
Creates 10 dummy 224x224 PNG images per class (normal/abnormal) in data/raw/.
Only used when real OpenPOCUS data is unavailable.

Usage:
  python scripts/generate_mock_data.py --output data/raw --n 10
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def generate_mock_images(output_dir: Path, n: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    classes = {"normal": 128, "abnormal": 80}

    for cls, base_brightness in classes.items():
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            # Simulate ultrasound-like grayscale noise image
            img_array = rng.integers(
                max(0, base_brightness - 40),
                min(255, base_brightness + 40),
                size=(224, 224),
                dtype=np.uint8,
            )
            noise = rng.normal(0, 15, size=(224, 224))
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode="L").convert("RGB")
            # Use PAT prefix so build_manifest extracts PAT{i:03d} as patient_id
            fname = f"PAT{i:03d}_frame01.png"
            img.save(cls_dir / fname)

        print(f"  {cls}: {n} mock images -> {cls_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n} mock images per class in {args.output}/")
    generate_mock_images(Path(args.output), n=args.n, seed=args.seed)
    print("Mock data ready.")


if __name__ == "__main__":
    main()
