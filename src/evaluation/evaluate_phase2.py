"""
Phase 2 Evaluation Script
Loads best Phase 2 checkpoint, runs on test split, outputs full 3-class diagnostic report.

Outputs saved to models/phase2/reports/:
  - confusion_matrix.png
  - roc_curves.png
  - phase2_eval_report.json
  - evaluation_summary.txt

Usage:
  python src/evaluation/evaluate_phase2.py --config config/phase2_config.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
CLASS_NAMES = ["Normal", "Moderate Risk", "High Risk"]


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
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "image_path": str(p),
                    "label":      label,
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


class EvalPhase2Dataset(Dataset):
    def __init__(self, image_records, clinical_df, split, split_map,
                 clinical_features, clin_mu, clin_std,
                 image_size=224, seed=42):
        self.rng      = random.Random(seed)
        self.features = clinical_features
        self.mu       = clin_mu.astype(np.float32)
        self.std      = clin_std.astype(np.float32)

        df = image_records.copy()
        df["split"] = df["patient_id"].map(split_map)
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.clin_by_class = {
            c: clinical_df[clinical_df["lus_class"] == c].reset_index(drop=True)
            for c in range(3)
        }
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        pool  = self.clin_by_class[label]
        clin_row = pool.iloc[self.rng.randint(0, len(pool) - 1)]
        raw  = np.array(clin_row[self.features].astype(float).values, dtype=np.float32)
        norm = (raw - self.mu) / self.std
        return {
            "image":    image,
            "clinical": torch.tensor(norm,  dtype=torch.float32),
            "label":    torch.tensor(label, dtype=torch.long),
        }


def run_inference(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            clin  = batch["clinical"].to(device)
            lbls  = batch["label"]
            out   = model(imgs, clin)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_labels.extend(lbls.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Phase 2 — Confusion Matrix (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(y_true, probs, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "forestgreen"]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        auc = roc_auc_score(y_bin, probs[:, i])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 2 — Per-Class ROC Curves (Test Set)", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/phase2_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()

    report_dir = Path(cfg["output"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                Path(cfg["output"]["checkpoint_dir"]) / "phase2_best.pth"

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    features  = list(cfg["clinical_features"])
    mu_dict   = ckpt.get("clinical_mu",  {})
    std_dict  = ckpt.get("clinical_std", {})
    clin_mu   = np.array([mu_dict[f]  for f in features], dtype=np.float32)
    clin_std  = np.array([std_dict[f] for f in features], dtype=np.float32)
    clin_std  = np.where(clin_std == 0, 1.0, clin_std)

    model = create_phase2_model(
        pretrained=False,
        freeze_backbone=False,
        clinical_input_dim=len(features),
        clinical_hidden_dim=cfg["model"]["clinical_hidden_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Best epoch: {ckpt.get('epoch', '?')}")

    image_root = Path("data/synthetic/neonatal_images")
    img_df     = build_image_records(image_root)
    clin_path  = Path(cfg["data"]["clinical_csv_path"])
    clin_df    = pd.read_csv(clin_path)

    seed      = cfg["training"]["random_seed"]
    split_map = patient_level_split(img_df["patient_id"].tolist(), seed=seed)

    test_ds = EvalPhase2Dataset(
        img_df, clin_df, "test", split_map, features,
        clin_mu, clin_std, cfg["data"]["image_size"], seed=seed,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    y_true, y_pred, probs = run_inference(model, test_loader, device)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc_ovr = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_ovr = None

    cr = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    print(f"\n{'='*55}")
    print("  PHASE 2 EVALUATION RESULTS (Test Split)")
    print(f"{'='*55}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    if auc_ovr is not None:
        print(f"  ROC-AUC (OvR):     {auc_ovr:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, report_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, probs, CLASS_NAMES, report_dir / "roc_curves.png")

    report = {
        "best_epoch":  ckpt.get("epoch"),
        "test_samples": int(len(y_true)),
        "label_distribution": {
            c: int((y_true == i).sum()) for i, c in enumerate(CLASS_NAMES)
        },
        "balanced_accuracy":     float(bal_acc),
        "roc_auc_ovr_macro":     auc_ovr,
        "classification_report": cr,
        "confusion_matrix":      confusion_matrix(y_true, y_pred).tolist(),
    }
    rpt_path = report_dir / "phase2_eval_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {rpt_path}")

    lines = [
        "Phase 2 Evaluation Summary",
        "=" * 45,
        f"Test set: {len(y_true)} samples",
        f"  Normal={int((y_true==0).sum())}  Moderate={int((y_true==1).sum())}  HighRisk={int((y_true==2).sum())}",
        f"Balanced Accuracy: {bal_acc:.4f}",
        (f"ROC-AUC (OvR):     {auc_ovr:.4f}" if auc_ovr else "ROC-AUC: N/A"),
        "",
        classification_report(y_true, y_pred, target_names=CLASS_NAMES),
    ]
    txt_path = report_dir / "evaluation_summary.txt"
    txt_path.write_text("\n".join(lines))
    print(f"  Text summary: {txt_path}")
    print(f"\nAll outputs: {report_dir}/")


if __name__ == "__main__":
    main()
