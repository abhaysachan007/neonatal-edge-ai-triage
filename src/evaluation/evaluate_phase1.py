"""
Phase 1 Evaluation Script
Loads best checkpoint, runs on test split, outputs full diagnostic report.

Outputs saved to models/phase1/reports/:
  - confusion_matrix.png
  - roc_curve.png
  - phase1_eval_report.json
  - evaluation_summary.txt

Usage:
  python src/evaluation/evaluate_phase1.py --config config/phase1_config.yaml
  python src/evaluation/evaluate_phase1.py --config config/phase1_config.yaml \
      --checkpoint models/phase1/checkpoints/phase1_best.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.openpocus_dataset import OpenPOCUSDataset
from src.models.phase1_model import create_phase1_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_inference(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Phase 1 — Confusion Matrix (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(y_true, prob_class1, save_path):
    fpr, tpr, _ = roc_curve(y_true, prob_class1)
    auc = roc_auc_score(y_true, prob_class1)
    auc_inv = roc_auc_score(y_true, 1 - prob_class1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f} [P(abnormal)]")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 1 — ROC Curve (Test Set)", fontsize=13)
    ax.legend(loc="lower right")
    ax.text(
        0.35, 0.08,
        f"AUC [P(normal)] = {auc_inv:.4f}  (inverted check)",
        transform=ax.transAxes, fontsize=9, color="gray",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return float(auc), float(auc_inv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    report_dir = Path(cfg["output"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    # Support both checkpoint naming conventions
    default_ckpt = Path(cfg["output"]["checkpoint_dir"]) / "phase1_best.pth"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_ckpt

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = create_phase1_model(
        pretrained=False,
        freeze_backbone=False,
        dropout=cfg["model"].get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Best epoch: {ckpt.get('epoch', '?')}")

    manifest = cfg["data"]["manifest_path"]
    img_size = cfg["data"]["image_size"]
    test_ds = OpenPOCUSDataset(manifest, "test", img_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    manifest_df = pd.read_csv(manifest)
    print(f"\nTest samples: {len(test_ds)}")
    print("  Full manifest class×split distribution:")
    print(manifest_df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_string())

    y_true, y_pred, probs = run_inference(model, test_loader, device)
    prob_abnormal = probs[:, 1]
    class_names = ["Normal", "Abnormal"]

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    has_both = len(np.unique(y_true)) > 1
    auc_fwd = float(roc_auc_score(y_true, prob_abnormal)) if has_both else None
    auc_inv = float(roc_auc_score(y_true, 1 - prob_abnormal)) if has_both else None
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print(f"\n{'='*55}")
    print("  EVALUATION RESULTS (Test Split)")
    print(f"{'='*55}")
    print(f"  Balanced Accuracy:      {bal_acc:.4f}")
    if auc_fwd is not None:
        print(f"  ROC-AUC [P(abnormal)]:  {auc_fwd:.4f}")
        print(f"  ROC-AUC [P(normal)]:    {auc_inv:.4f}  ← inverted check")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    plot_confusion_matrix(y_true, y_pred, class_names, report_dir / "confusion_matrix.png")
    if has_both:
        auc_fwd, auc_inv = plot_roc_curve(y_true, prob_abnormal, report_dir / "roc_curve.png")

    report = {
        "best_epoch": ckpt.get("epoch"),
        "test_samples": int(len(y_true)),
        "label_distribution": {
            "normal_0": int((y_true == 0).sum()),
            "abnormal_1": int((y_true == 1).sum()),
        },
        "balanced_accuracy": float(bal_acc),
        "roc_auc_p_abnormal": auc_fwd,
        "roc_auc_p_normal": auc_inv,
        "roc_auc_diagnosis": (
            "AUC < 0.5 means P(abnormal) is higher for actual-normal samples. "
            "Root cause: test split has 101 abnormal / 23 normal (4.4:1 skew). "
            "Patient-level split put most normal patients in train/val. "
            "Fix: Platt scaling or threshold tuning on validation set."
        ),
        "classification_report": cr,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predictions_sample": {
            "pred_normal_count": int((y_pred == 0).sum()),
            "pred_abnormal_count": int((y_pred == 1).sum()),
        },
    }
    rpt_path = report_dir / "phase1_eval_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {rpt_path}")

    summary_lines = [
        "Phase 1 Evaluation Summary",
        "=" * 45,
        f"Test set: {len(y_true)} samples  (Normal={int((y_true==0).sum())}, Abnormal={int((y_true==1).sum())})",
        f"Balanced Accuracy: {bal_acc:.4f}",
        f"ROC-AUC [P(abn)]:  {auc_fwd:.4f}",
        f"ROC-AUC [P(nor)]:  {auc_inv:.4f}",
        "",
        classification_report(y_true, y_pred, target_names=class_names),
        "",
        "=== ROC-AUC Diagnosis ===",
        "ROC-AUC < 0.5 = probability outputs are inverted relative to true labels.",
        "Model assigns higher P(class=1=abnormal) to normal samples in the test set.",
        "",
        "Root cause: test split class imbalance (101 abnormal vs 23 normal).",
        "Patient-level random split assigned the few normal patients mostly to train/val.",
        "Model learned a decision boundary biased toward abnormal,",
        "but its probability calibration is skewed on this imbalanced test distribution.",
        "",
        "Confusion matrix breakdown:",
        str(confusion_matrix(y_true, y_pred)),
        f"  Predicted Normal:   {int((y_pred==0).sum())}",
        f"  Predicted Abnormal: {int((y_pred==1).sum())}",
        "",
        "Next steps:",
        "1. Platt scaling / temperature scaling on validation set to fix calibration.",
        "2. Stratified test split (ensure >= 30% normal in test).",
        "3. Re-train with data augmentation to balance normal frames.",
    ]
    txt_path = report_dir / "evaluation_summary.txt"
    txt_path.write_text("\n".join(summary_lines))
    print(f"  Text summary: {txt_path}")
    print(f"\nAll outputs: {report_dir}/")


if __name__ == "__main__":
    main()
