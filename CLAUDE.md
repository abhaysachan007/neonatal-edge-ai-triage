# CLAUDE.md — Neonatal LUS Synthetic AI Project
## Context for any Claude session working on this repo

---

## What this project is

A **from-scratch neonatal lung ultrasound (LUS) AI triage system** built entirely on:
- **Publicly available adult LUS data** (OpenPOCUS dataset — free download)
- **Synthetically generated neonatal-domain images** (adult LUS → domain-adapted via augmentation pipeline)
- **Synthetically generated clinical tabular data** (sampled from published neonatal RDS cohort statistics)

**No real neonatal patient data is used anywhere in this project.**

---

## Why synthetic data?

Real neonatal LUS datasets are scarce, proprietary, or ethically restricted.
This project demonstrates that a clinically-meaningful triage AI can be developed using:
1. Adult LUS images as a base (OpenPOCUS — publicly licensed)
2. Domain adaptation transforms that simulate neonatal ultrasound characteristics
3. Synthetic clinical vitals generated from published RDS literature distributions

---

## Project phases

### Phase 1 — Minor (Binary classifier on adult LUS)
- Task: Binary classification — Normal vs Abnormal LUS
- Data: OpenPOCUS adult lung ultrasound frames
- Model: MobileNetV3-Small (edge-deployable, <2MB)
- Output: Trained `.pth` checkpoint + evaluation report
- Status: **Build this first**

### Phase 2 — Major (Neonatal triage with synthetic data)
- Task: 3-class triage — Normal / Moderate Risk / High Risk
- Data: Synthetic neonatal LUS (from adult base) + synthetic clinical CSV
- Model: MobileNetV3-Small + lightweight clinical MLP → late fusion
- Output: Multimodal triage prediction with confidence score
- Status: **Build after Phase 1 is working**

---

## Data sources

| Source | What | Where |
|--------|------|--------|
| OpenPOCUS | Adult LUS frames (Normal/Abnormal) | https://openpocus.com / Zenodo |
| Synthetic neonatal images | Generated from OpenPOCUS via `src/data/neonatal_synthesizer.py` | Local |
| Synthetic clinical CSV | Generated via `scripts/generate_clinical_data.py` | Local |

---

## Tech stack

- Python 3.10+
- PyTorch + torchvision
- scikit-learn, pandas, numpy
- OpenCV (image augmentation)
- Pillow
- matplotlib (plots)
- CTGAN (optional, for tabular synthesis — fallback: numpy sampling)

---

## Folder structure

```
neonatal-edge-ai-triage/
├── CLAUDE.md                     ← YOU ARE HERE
├── README.md
├── requirements.txt
├── config/
│   ├── phase1_config.yaml        ← Phase 1 training config
│   └── phase2_config.yaml        ← Phase 2 training config
├── src/
│   ├── data/
│   │   ├── openpocus_dataset.py  ← Phase 1 dataset loader
│   │   ├── neonatal_synthesizer.py ← Adult→Neonatal image transforms
│   │   ├── neonatal_dataset.py   ← Phase 2 dataset loader
│   │   └── clinical_dataset.py   ← Clinical tabular data loader
│   ├── models/
│   │   ├── phase1_model.py       ← MobileNetV3 binary classifier
│   │   └── phase2_model.py       ← MobileNetV3 + MLP fusion model
│   ├── training/
│   │   ├── train_phase1.py       ← Phase 1 training loop
│   │   └── train_phase2.py       ← Phase 2 training loop
│   ├── evaluation/
│   │   ├── evaluate_phase1.py    ← Binary eval metrics
│   │   └── evaluate_phase2.py    ← 3-class eval + calibration
│   └── inference/
│       └── predict.py            ← Single image + clinical → triage output
├── scripts/
│   ├── download_openpocus.py     ← Instructions + auto-download helper
│   ├── build_manifest.py         ← Build train/val/test CSV manifest
│   ├── generate_clinical_data.py ← Synthetic clinical CSV generator
│   └── generate_neonatal_images.py ← Batch synthesize neonatal images
├── data/
│   ├── raw/                      ← OpenPOCUS raw frames go here
│   ├── synthetic/                ← Generated neonatal images go here
│   └── processed/                ← Manifests, splits CSVs
└── docs/
    ├── architecture.md
    ├── dataset_cards.md
    └── experimental_log.md
```

---

## Key design decisions

1. **Binary in Phase 1, 3-class in Phase 2** — keeps Phase 1 achievable and evaluable
2. **MobileNetV3-Small always** — edge deployment constraint (Raspberry Pi / Jetson Nano target)
3. **Patient-level splits** — no data leakage (one patient's frames stay in one split)
4. **ImageNet normalization** — standard mean/std throughout
5. **Weighted loss** — handles class imbalance in both phases
6. **Synthetic neonatal transforms**: brightness↑, contrast↑, thin pleural line simulation, smaller FOV crop, speckle noise

---

## How to run (quick start)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Download OpenPOCUS data
python scripts/download_openpocus.py

# 3. Build manifest
python scripts/build_manifest.py

# 4. Train Phase 1
python src/training/train_phase1.py --config config/phase1_config.yaml

# 5. Evaluate Phase 1
python src/evaluation/evaluate_phase1.py --checkpoint models/phase1_best.pth

# 6. Generate synthetic neonatal images (for Phase 2)
python scripts/generate_neonatal_images.py

# 7. Generate synthetic clinical data
python scripts/generate_clinical_data.py

# 8. Train Phase 2
python src/training/train_phase2.py --config config/phase2_config.yaml
```

---

## What NOT to do

- Do NOT add real patient data anywhere in this repo
- Do NOT hardcode local file paths — use config files
- Do NOT skip patient-level splits — frame-level split = data leakage
- Do NOT use BatchNorm during single-image inference — set model.eval() always
- Do NOT confuse Phase 1 (binary) with Phase 2 (3-class) — separate configs, separate models

---

## If you are a Claude session starting fresh

1. Read this file first (done)
2. Check `docs/experimental_log.md` for current status
3. Check which phase is being worked on
4. Use config YAMLs — never hardcode hyperparameters
5. Ask before making architectural changes to models
