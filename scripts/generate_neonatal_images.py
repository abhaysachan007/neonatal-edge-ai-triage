"""
Generate Synthetic Neonatal LUS Images from OpenPOCUS Adult Data

Maps adult binary classes → neonatal 3 classes:
  normal   (adult) → class_0_normal    (severity=0)
  abnormal (adult) → class_1_moderate  (severity=1)  [50%]
                  → class_2_high_risk  (severity=2)  [50%]

Usage:
  python scripts/generate_neonatal_images.py \
      --input data/raw \
      --output data/synthetic/neonatal_images
"""

import argparse
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.neonatal_synthesizer import NeonatalSynthesizer

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/synthetic/neonatal_images")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    synth = NeonatalSynthesizer(seed=args.seed)
    rng = random.Random(args.seed)

    # Output class dirs
    class_dirs = {
        0: output_dir / "class_0_normal",
        1: output_dir / "class_1_moderate",
        2: output_dir / "class_2_high_risk",
    }
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    counts = {0: 0, 1: 0, 2: 0}

    for class_name in ["normal", "abnormal"]:
        src = input_dir / class_name
        if not src.exists():
            print(f"[WARN] Not found: {src}")
            continue

        for img_path in sorted(src.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")

                if class_name == "normal":
                    # Normal adult → Normal neonatal
                    neo_label = 0
                    severity = 0
                else:
                    # Abnormal adult → Moderate (50%) or High Risk (50%)
                    neo_label = rng.choice([1, 2])
                    severity = neo_label

                synth_img = synth.synthesize_pil(pil_img, severity=severity)
                out_path = class_dirs[neo_label] / img_path.name
                synth_img.save(out_path)
                counts[neo_label] += 1

            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")

    print(f"\nSynthetic neonatal images generated → {output_dir}")
    print(f"  Normal (0):       {counts[0]:,}")
    print(f"  Moderate (1):     {counts[1]:,}")
    print(f"  High Risk (2):    {counts[2]:,}")
    print(f"  Total:            {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
