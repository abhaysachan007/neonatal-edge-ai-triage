"""
Phase 1 Training Script
Binary LUS classification on OpenPOCUS adult data

Usage:
  python src/training/train_phase1.py --config config/phase1_config.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.openpocus_dataset import OpenPOCUSDataset
from src.models.phase1_model import create_phase1_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y, p, pr = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "f1": f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score(y, pr) if len(np.unique(y)) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    # Output dirs
    for key in ["checkpoint_dir", "report_dir", "figure_dir", "history_dir"]:
        Path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)

    # Seed
    seed = cfg["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*50}")
    print("  PHASE 1 TRAINING — Binary LUS Classifier")
    print(f"{'='*50}")
    print(f"Device: {device}")

    # Datasets
    manifest = cfg["data"]["manifest_path"]
    img_size = cfg["data"]["image_size"]
    workers = cfg["data"]["num_workers"]

    train_ds = OpenPOCUSDataset(manifest, "train", img_size, train=True)
    val_ds   = OpenPOCUSDataset(manifest, "validation", img_size, train=False)
    test_ds  = OpenPOCUSDataset(manifest, "test", img_size, train=False)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=workers)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # Class weights
    labels = train_ds.df["label"].astype(int).values
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
    print(f"Class weights: {weights.cpu().numpy()}")

    # Model
    model = create_phase1_model(
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg["training"]["scheduler_factor"],
        patience=cfg["training"]["scheduler_patience"],
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / "phase1_best.pth"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels_b = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels_b).sum().item()
            total += images.size(0)

        train_loss = ep_loss / total
        train_acc  = correct / total
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.perf_counter() - t0,
        })

        print(f"\nEpoch {epoch:02d}/{cfg['training']['epochs']} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_metrics['loss']:.4f} | "
              f"ValAcc={val_metrics['accuracy']:.4f} | ValF1={val_metrics['f1']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics, "config": cfg}, ckpt_path)
            print("  ✓ Checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print("\nEarly stopping.")
                break

    # Save history
    pd.DataFrame(history).to_csv(
        Path(cfg["output"]["history_dir"]) / "phase1_history.csv", index=False
    )

    # Final test evaluation
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"  FINAL TEST (best epoch: {best_epoch})")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    # Save report
    report = {"phase": 1, "best_epoch": best_epoch, "test_metrics": test_metrics, "config": cfg}
    with open(Path(cfg["output"]["report_dir"]) / "phase1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
