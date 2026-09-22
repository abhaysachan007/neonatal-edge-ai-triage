"""
Neonatal LUS Image Synthesizer
================================
Converts adult OpenPOCUS LUS frames into synthetic neonatal-domain images.

Neonatal vs Adult LUS differences simulated:
  1. Higher gain → brighter image overall
  2. Thinner chest wall → crop top 15% (less subcutaneous tissue)
  3. Smaller field of view → center crop to 80% then resize
  4. B-lines more prominent → contrast boost in lower half
  5. Pleural line thinner/shallower → slight blur on top band
  6. Higher frequency probe speckle → fine noise texture
  7. Label mapping: adult Normal→Neonatal Normal(0), adult Abnormal→ risk level via severity score
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random


class NeonatalSynthesizer:
    """
    Apply deterministic + stochastic transforms to simulate neonatal LUS appearance.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def synthesize(self, image: np.ndarray, severity: int = 0) -> np.ndarray:
        """
        Args:
            image:    HxWx3 uint8 numpy array (RGB)
            severity: 0=Normal, 1=Moderate, 2=High — controls augmentation intensity

        Returns:
            Synthesized image as HxWx3 uint8 numpy array (RGB)
        """
        img = image.copy().astype(np.float32)
        h, w = img.shape[:2]

        # ── 1. Crop top 15% (thin chest wall simulation) ──
        top_crop = int(h * 0.15)
        img = img[top_crop:, :, :]
        img = cv2.resize(img, (w, h))  # restore size

        # ── 2. Smaller FOV — center crop 80% ──
        crop_h = int(h * 0.80)
        crop_w = int(w * 0.80)
        y_start = (h - crop_h) // 2
        x_start = (w - crop_w) // 2
        img = img[y_start:y_start+crop_h, x_start:x_start+crop_w, :]
        img = cv2.resize(img, (w, h))

        # ── 3. Higher gain (brighter overall) ──
        gain = 1.2 + self.rng.uniform(0, 0.15)
        img = np.clip(img * gain, 0, 255)

        # ── 4. B-line enhancement in lower 60% (severity-dependent) ──
        lower_start = int(h * 0.40)
        bline_boost = 1.0 + (0.15 * severity) + self.rng.uniform(0, 0.05)
        img[lower_start:, :, :] = np.clip(
            img[lower_start:, :, :] * bline_boost, 0, 255
        )

        # ── 5. Pleural line blur (thin/shallow simulation) ──
        pleural_band = int(h * 0.12)
        img[:pleural_band, :, :] = cv2.GaussianBlur(
            img[:pleural_band, :, :].astype(np.uint8),
            (5, 5), 1.0
        ).astype(np.float32)

        # ── 6. High-freq speckle noise (simulates higher MHz probe) ──
        noise_std = 8 + (3 * severity)
        noise = self.rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)

        # ── 7. Slight contrast stretch ──
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0

        return img.astype(np.uint8)

    def synthesize_pil(self, pil_image: Image.Image, severity: int = 0) -> Image.Image:
        """Convenience wrapper for PIL images."""
        arr = np.array(pil_image.convert("RGB"))
        synth = self.synthesize(arr, severity=severity)
        return Image.fromarray(synth)


def batch_synthesize(
    input_dir: str,
    output_dir: str,
    label_map: dict,
    seed: int = 42,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
):
    """
    Batch convert a directory of adult LUS images to synthetic neonatal images.

    Args:
        input_dir:  path to folder with adult LUS images (subfolders = class labels)
        output_dir: where to save synthetic images
        label_map:  dict mapping adult subfolder name → (neonatal_label, severity)
                    e.g. {"normal": (0, 0), "abnormal": (2, 2)}
        seed:       random seed
        extensions: image file extensions to process

    Output structure:
        output_dir/
            class_0_normal/
            class_1_moderate/
            class_2_high_risk/
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    synth = NeonatalSynthesizer(seed=seed)

    class_dirs = {
        0: output_dir / "class_0_normal",
        1: output_dir / "class_1_moderate",
        2: output_dir / "class_2_high_risk",
    }
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    count = 0
    for subfolder, (neo_label, severity) in label_map.items():
        src_folder = input_dir / subfolder
        if not src_folder.exists():
            print(f"[WARN] Subfolder not found: {src_folder}")
            continue

        for img_path in src_folder.iterdir():
            if img_path.suffix.lower() not in extensions:
                continue
            try:
                pil_img = Image.open(img_path).convert("RGB")
                synth_img = synth.synthesize_pil(pil_img, severity=severity)
                out_path = class_dirs[neo_label] / img_path.name
                synth_img.save(out_path)
                count += 1
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

    print(f"\nDone. {count} images synthesized → {output_dir}")
    return count


if __name__ == "__main__":
    # Quick test — synthesize one image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python neonatal_synthesizer.py <image_path> [severity 0/1/2]")
        sys.exit(1)

    img_path = sys.argv[1]
    severity = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    pil_img = Image.open(img_path).convert("RGB")
    synth = NeonatalSynthesizer()
    result = synth.synthesize_pil(pil_img, severity=severity)
    out = Path(img_path).stem + f"_neonatal_sev{severity}.png"
    result.save(out)
    print(f"Saved: {out}")
