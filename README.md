# Neonatal LUS Edge AI Triage
### Synthetic Neonatal Lung Ultrasound Classification System

> **No real neonatal patient data used.** All neonatal images are synthetically derived from publicly available adult LUS data (OpenPOCUS, CC-BY 4.0). Clinical data is generated from published RDS cohort statistics.

---

## What this is

A two-phase AI pipeline for neonatal respiratory triage using lung ultrasound (LUS):

- **Phase 1** — Binary classifier (Normal vs Abnormal) trained on adult OpenPOCUS LUS data
- **Phase 2** — 3-class multimodal triage model (Normal / Moderate Risk / High Risk) using synthetic neonatal LUS images + synthetic clinical vitals

Built for edge deployment — MobileNetV3-Small backbone (~1.5M parameters).

---

## Quick Start

```bash
pip install -r requirements.txt

# Build manifest from OpenPOCUS data
python scripts/build_manifest.py --input data/raw --output data/processed/phase1_manifest.csv

# Train Phase 1
python src/training/train_phase1.py --config config/phase1_config.yaml

# Generate synthetic neonatal images for Phase 2
python scripts/generate_neonatal_images.py

# Generate synthetic clinical data
python scripts/generate_clinical_data.py

# Train Phase 2
python src/training/train_phase2.py --config config/phase2_config.yaml
```

---

## Data

| Dataset | Type | Source |
|---------|------|--------|
| OpenPOCUS | Real adult LUS | https://zenodo.org/communities/openpocus |
| Synthetic neonatal LUS | Derived from OpenPOCUS | `src/data/neonatal_synthesizer.py` |
| Synthetic clinical | Generated from literature | `scripts/generate_clinical_data.py` |

---

## Project Structure

```
├── CLAUDE.md                    ← AI session context doc
├── config/
│   ├── phase1_config.yaml
│   └── phase2_config.yaml
├── src/
│   ├── data/
│   │   ├── openpocus_dataset.py
│   │   ├── neonatal_synthesizer.py
│   │   └── ...
│   ├── models/
│   │   ├── phase1_model.py
│   │   └── phase2_model.py
│   └── training/
│       ├── train_phase1.py
│       └── train_phase2.py
├── scripts/
│   ├── build_manifest.py
│   ├── generate_clinical_data.py
│   └── generate_neonatal_images.py
└── docs/
    └── experimental_log.md
```

---

## Model Architecture

**Phase 1:** MobileNetV3-Small → Binary head (2 classes)

**Phase 2:**
```
Image → MobileNetV3-Small encoder (576-dim)
                                              → Concat (608-dim) → FC → 3 classes
Clinical → MLP encoder (8→32→32-dim)
```

---

## Ethical Note

This system is a **research prototype**. Synthetic data means:
- Results cannot be directly extrapolated to clinical performance
- Model must be validated on real neonatal data before any clinical use
- Not a medical device
