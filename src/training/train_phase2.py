"""
Phase 2 Training Script
Multimodal triage: synthetic neonatal LUS images + synthetic clinical data → 3-class

Classes: 0=Normal, 1=Moderate Risk, 2=High Risk

Image source:  data/synthetic/neonatal_images/class_0_normal/
                                               class_1_moderate/
                                               class_2_high_risk/
Clinical CSV:  data/synthetic/clinical_data.csv

Pairing strategy (synthetic data):
  For each image (class C), randomly assign a clinical record of class C.
  Patient-level splits derived from image filename prefix (first 2 parts).

Usage:
  python src/training/train_phase2.py --config config/phase2_config.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.phase2_model import create_phase2_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".bmp"}

CLASS_DIRS = {
    0: "class_0_normal",
    1: "class_1_moderate",
    2: "class_2_high_risk",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patient_id_from_path(img_path: Path) -> str:
    parts = img_path.stem.split("_")
    return "_".join(parts[:2]).upper() if len(parts) >= 2 else img_path.stem.upper()


def build_image_records(image_root: Path) -> pd.DataFrame:
    rows = []
    for label, dirname in CLASS_DIRS.items():
        class_dir = image_root / dirname
        if not class_dir.exists():
            print(f"[WARN] Not found: {class_dir}")
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "image_path": str(p),
                    "label": label,
                    "patient_id": patient_id_from_path(p),
                })
    return pd.DataFrame(rows)


def patient_level_split(patient_ids, train_frac=0.70, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    unique = sorted(set(patient_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    split_map = {}
    for i, pid in enumerate(unique):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "validation"
        else:
            split_map[pid] = "test"
    return split_map


class Phase2Dataset(Dataset):
    """
    Pairs each synthetic neonatal image with a random clinical record
    of the same class. Both are synthetic, so random pairing is valid.
    """

    def __init__(
        self,
        image_records: pd.DataFrame,
        clinical_df: pd.DataFrame,
        split: str,
        split_map: dict,
        clinical_features: list,
        clinical_mu: pd.Series,
        clinical_std: pd.Series,
        image_size: int = 224,
        train: bool = False,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.features = clinical_features
        self.mu = clinical_mu.values.astype(np.float32)
        self.std = clinical_std.values.astype(np.float32)

        df = image_records.copy()
        df["split"] = df["patient_id"].map(split_map)
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.clin_by_class = {
            c: clinical_df[clinical_df["lus_class"] == c].reset_index(drop=True)
            for c in range(3)
        }

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])
        pool = self.clin_by_class[label]
        clin_row = pool.iloc[self.rng.randint(0, len(pool) - 1)]
        raw = np.array(clin_row[self.features].astype(float).values, dtype=np.float32)
        normalized = (raw - self.mu) / self.std

        return {
            "image": image,
            "clinical": torch.tensor(normalized, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate_phase2(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            lbls = batch["label"].to(device)
            out  = model(imgs, clin)
            loss = criterion(out, lbls)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            all_labels.extend(lbls.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y  = np.array(all_labels)
    p  = np.array(all_preds)
    pr = np.vstack(all_probs)

    metrics = {
        "loss":               total_loss / len(loader.dataset),
        "accuracy":           float(accuracy_score(y, p)),
        "balanced_accuracy":  float(balanced_accuracy_score(y, p)),
        "f1_macro":           float(f1_score(y, p, average="macro",    zero_division=0)),
        "f1_weighted":        float(f1_score(y, p, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y, pr, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc_ovr"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()
    seed   = cfg["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    for key in ["checkpoint_dir", "report_dir", "figure_dir", "history_dir"]:
        Path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  PHASE 2 TRAINING — Neonatal Triage (3-class)")
    print(f"{'='*55}")
    print(f"Device: {device}")

    # ── Images ──
    image_root = Path("data/synthetic/neonatal_images")
    img_df = build_image_records(image_root)
    if len(img_df) == 0:
        print("[ERROR] No synthetic neonatal images found.")
        print("  Run: python scripts/generate_neonatal_images.py --input data/raw")
        return
    print(f"Images found: {len(img_df)}")
    print(f"  {img_df['label'].value_counts().sort_index().to_dict()}")

    # ── Clinical CSV ──
    clin_path = Path(cfg["data"]["clinical_csv_path"])
    if not clin_path.exists():
        print(f"[ERROR] Clinical CSV not found: {clin_path}")
        print("  Run: python scripts/generate_clinical_data.py")
        return
    clin_df  = pd.read_csv(clin_path)
    features = list(cfg["clinical_features"])
    print(f"Clinical records: {len(clin_df)}")

    # ── Splits ──
    split_map = patient_level_split(img_df["patient_id"].tolist(), seed=seed)
    img_df["split"] = img_df["patient_id"].map(split_map)
    print("\nImage split distribution:")
    print(img_df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    # Normalisation stats
    mu  = clin_df[features].mean()
    std = clin_df[features].std().replace(0, 1.0)

    img_size = cfg["data"]["image_size"]
    bs       = cfg["training"]["batch_size"]

    train_ds = Phase2Dataset(img_df, clin_df, "train",      split_map, features, mu, std, img_size, train=True,  seed=seed)
    val_ds   = Phase2Dataset(img_df, clin_df, "validation", split_map, features, mu, std, img_size, train=False, seed=seed)
    test_ds  = Phase2Dataset(img_df, clin_df, "test",       split_map, features, mu, std, img_size, train=False, seed=seed)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    # ── Class weights ──
    train_labels = img_df[img_df["split"] == "train"]["label"].astype(int).values
    counts  = np.bincount(train_labels, minlength=3)
    weights = torch.tensor(
        counts.sum() / (3.0 * (counts + 1e-6)), dtype=torch.float32
    ).to(device)
    print(f"Class weights: {weights.cpu().numpy()}")

    # ── Model ──
    model = create_phase2_model(
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        clinical_input_dim=len(features),
        clinical_hidden_dim=cfg["model"]["clinical_hidden_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    # ── Training loop ──
    ckpt_path    = Path(cfg["output"]["checkpoint_dir"]) / "phase2_best.pth"
    best_val_loss = float("inf")
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        for batch in train_loader:
            imgs = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out  = model(imgs, clin)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            ep_loss  += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == lbls).sum().item()
            total    += imgs.size(0)

        train_loss = ep_loss / total
        train_acc  = correct / total
        val_m      = evaluate_phase2(model, val_loader, criterion, device)

        if cfg["training"]["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_m["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            **{f"val_{k}": v for k, v in val_m.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.perf_counter() - t0,
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_m['loss']:.4f} | "
              f"ValAcc={val_m['accuracy']:.4f} | ValF1={val_m['f1_macro']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "config": cfg,
                "clinical_mu": mu.to_dict(),
                "clinical_std": std.to_dict(),
            }, ckpt_path)
            print("  ✓ Checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["training"]["early_stopping_patience"]:
                print("\nEarly stopping.")
                break

    # ── Save history ──
    hist_path = Path(cfg["output"]["history_dir"]) / "phase2_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"\nHistory: {hist_path}")

    # ── Final test ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_phase2(model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"  FINAL TEST (best epoch: {best_epoch})")
    print(f"{'='*55}")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    rpt = {
        "phase": 2,
        "best_epoch": best_epoch,
        "test_metrics": test_m,
        "train_samples": int(len(train_ds)),
        "val_samples":   int(len(val_ds)),
        "test_samples":  int(len(test_ds)),
    }
    rpt_path = Path(cfg["output"]["report_dir"]) / "phase2_report.json"
    with open(rpt_path, "w") as f:
        json.dump(rpt, f, indent=2)

    print(f"\nDone. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
