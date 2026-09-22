"""
Export Phase 1 and Phase 2 models as TorchScript for edge deployment.

Outputs:
  models/phase1/phase1_traced.pt
  models/phase2/phase2_traced.pt

Usage:
  python scripts/export_models.py
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.phase1_model import create_phase1_model
from src.models.phase2_model import create_phase2_model


def mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def export_phase1(ckpt_path: Path, out_path: Path, cfg: dict):
    print(f"\n[Phase 1] Loading: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_phase1_model(
        pretrained=False,
        freeze_backbone=False,
        dropout=cfg["model"].get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    traced = torch.jit.trace(model, torch.zeros(1, 3, 224, 224))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    size = mb(out_path)
    print(f"  Saved: {out_path}  ({size:.2f} MB)")
    return size


def export_phase2(ckpt_path: Path, out_path: Path, cfg: dict):
    print(f"\n[Phase 2] Loading: {ckpt_path}")
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    features = list(cfg["clinical_features"])
    model    = create_phase2_model(
        pretrained=False,
        freeze_backbone=False,
        clinical_input_dim=len(features),
        clinical_hidden_dim=cfg["model"]["clinical_hidden_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    traced = torch.jit.trace(
        model,
        (torch.zeros(1, 3, 224, 224), torch.zeros(1, len(features))),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    size = mb(out_path)
    print(f"  Saved: {out_path}  ({size:.2f} MB)")
    return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt",   default="models/phase1/checkpoints/phase1_best.pth")
    parser.add_argument("--phase2-ckpt",   default="models/phase2/checkpoints/phase2_best.pth")
    parser.add_argument("--phase1-config", default="config/phase1_config.yaml")
    parser.add_argument("--phase2-config", default="config/phase2_config.yaml")
    args = parser.parse_args()

    with open(args.phase1_config) as f:
        cfg1 = yaml.safe_load(f)
    with open(args.phase2_config) as f:
        cfg2 = yaml.safe_load(f)

    size1 = export_phase1(Path(args.phase1_ckpt), Path("models/phase1/phase1_traced.pt"), cfg1)
    size2 = export_phase2(Path(args.phase2_ckpt), Path("models/phase2/phase2_traced.pt"), cfg2)

    print(f"\n{'='*45}")
    print("  EXPORT SUMMARY")
    print(f"{'='*45}")
    print(f"  Phase 1 TorchScript: models/phase1/phase1_traced.pt  ({size1:.2f} MB)")
    print(f"  Phase 2 TorchScript: models/phase2/phase2_traced.pt  ({size2:.2f} MB)")
    print(f"  Total: {size1 + size2:.2f} MB")


if __name__ == "__main__":
    main()
