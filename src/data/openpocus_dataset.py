"""
Phase 1 Dataset — OpenPOCUS Adult LUS
Binary classification: 0=Normal, 1=Abnormal

Expects a manifest CSV with columns:
  - frame_path: absolute or relative path to image
  - label: 0 or 1
  - patient_id: for patient-level splitting
  - split: train / validation / test
"""

from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class OpenPOCUSDataset(Dataset):
    """
    Loads OpenPOCUS adult LUS frames from a CSV manifest.
    """

    def __init__(self, manifest_path: str, split: str, image_size: int = 224, train: bool = False):
        assert split in {"train", "validation", "test"}, \
            f"split must be train/validation/test, got: {split}"

        df = pd.read_csv(manifest_path, low_memory=False)
        df["split"] = df["split"].astype(str).str.strip()
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

        self.transform = self._build_transform(image_size, train)

    def _build_transform(self, image_size: int, train: bool):
        if train:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["frame_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": str(row["patient_id"]),
        }
