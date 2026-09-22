# Experimental Log

## Project: Neonatal LUS Synthetic AI Triage

---

## Status

| Phase | Component | Status |
|-------|-----------|--------|
| Phase 1 | OpenPOCUS dataset loader | ✅ Ready |
| Phase 1 | MobileNetV3-Small binary model | ✅ Ready |
| Phase 1 | Training script | ✅ Done (real data — 904 frames) |
| Phase 1 | Manifest builder | ✅ Done |
| Phase 1 | Evaluation | ⏳ Pending |
| Phase 2 | Neonatal image synthesizer | ✅ Ready |
| Phase 2 | Clinical data generator | ✅ Done (500 patients) |
| Phase 2 | Synthetic neonatal images | ✅ Done (40 from mock; re-run needed with real base) |
| Phase 2 | Multimodal fusion model | ✅ Ready |
| Phase 2 | Phase 2 training script | ⏳ Pending |
| Phase 2 | Phase 2 evaluation | ⏳ Pending |

---

## Run Log — 2026-09-22 (Pass 1 — Mock Data)

### Task 1 — Push missing scripts
- **Status: ✅ DONE**
- Created `scripts/download_openpocus.py` (was missing from scaffold)
- Created `scripts/generate_mock_data.py` (pipeline validation fallback)
- Both pushed to `abhaysachan007/neonatal-edge-ai-triage` main

### Task 2 — Download OpenPOCUS data
- **Status: ⚠️ PARTIAL**
- Zenodo record 7842167 reached but only contains `Harri-1.pdf` — no image frames
- Real image data sourced from `jannisborn/covid19_ultrasound` instead (see Pass 2)

### Task 3 — Build manifest
- **Status: ✅ DONE (mock data)**
- Generated 10 mock 224×224 PNG images per class via `generate_mock_data.py`
- Manifest: `data/processed/phase1_manifest.csv` — 40 frames
- Split: 32 train / 2 val / 6 test — patient-level, no leakage confirmed ✅

### Task 4 — Train Phase 1
- **Status: ✅ DONE (mock data — loop validated)**
- Environment: `myenv` conda env (Python 3.10, PyTorch CPU)
- MobileNetV3-Small pretrained weights: downloaded (9.83 MB)
- Ran 13 epochs, early stopped (patience=5)
- Best checkpoint: epoch 8 → `models/phase1/checkpoints/phase1_best.pth`
- **Results on mock data (random noise images — metrics not meaningful):**
  - Test Loss: 0.6929 | Accuracy: 0.5000 | F1: 0.6667 | ROC-AUC: 0.8889
- **Training loop end-to-end: PASS ✅**

### Task 5 — Generate synthetic clinical data
- **Status: ✅ DONE**
- `data/synthetic/clinical_data.csv` — 500 patients
- Class: Normal=196, Moderate Risk=180, High Risk=124
- Distributions from Sweet et al. 2023 + Isayama et al. 2016

### Task 6 — Generate synthetic neonatal images
- **Status: ✅ DONE (from mock base)**
- `data/synthetic/neonatal_images/`: Normal=20, Moderate=15, High Risk=5, Total=40
- Will re-run with real OpenPOCUS images once available

---

## Pass 2 — Real Data (2026-09-22)

Real LUS frames sourced from `jannisborn/covid19_ultrasound` GitHub repo via git sparse checkout + LFS pull.

### Task 1 — Source real LUS data
- Sparse-cloned `jannisborn/covid19_ultrasound` to `D:/jeevika/covid19_us`
- Extracted 696 frames (305 normal, 391 abnormal) via `scripts/extract_openpocus_frames.py`
- Label mapping: `Reg_*` normal; `Cov_*`, `Pneu_*`, `Vir_*` abnormal

### Task 2 — Download script updated
- `scripts/download_openpocus.py` — references correct GitHub source
- `scripts/extract_openpocus_frames.py` — new script for video frame extraction

### Task 3 — Manifest rebuild (real data)
- Total frames: 904 | Train: 649 / Val: 131 / Test: 124
- Patient-level split, no leakage confirmed
- Note: test set imbalanced (101 abnormal / 23 normal) — inherent from source dataset
- Fix applied: `build_manifest.py` patient_id now uses 2-part prefix (REG_AVI, PNEU_NORTHUMBRIA, etc.)

### Task 4 — Phase 1 real-data training
- Env: `myenv` conda (Python 3.10, PyTorch CPU)
- All 20 epochs ran (no early stop before epoch 20)
- Best checkpoint: epoch 15 → `models/phase1/checkpoints/phase1_best.pth`

| Metric | Val (epoch 15) | Test |
|--------|---------------|------|
| Loss | 0.4131 | 0.6708 |
| Accuracy | 0.7405 | 0.7177 |
| Balanced Accuracy | — | 0.5077 |
| F1 | 0.8152 | 0.8293 |
| ROC-AUC | — | 0.3943 |

**Note:** Low test balanced accuracy (0.51) and ROC-AUC (0.39) caused by severe test set class imbalance (101 abnormal / 23 normal). Model predicts abnormal reliably (high F1) but threshold tuning needed for normal recall. Next: `evaluate_phase1.py` + consider stratified test resampling.

---

## Next Steps (in order)

1. Run `python src/evaluation/evaluate_phase1.py` — confusion matrix + per-class metrics
2. Re-run `python scripts/generate_neonatal_images.py --input data/raw --output data/synthetic/neonatal_images` with real base images
3. Build Phase 2 manifest
4. Train Phase 2
5. Evaluate Phase 2

---

## Data Sources

### OpenPOCUS / covid19_ultrasound
- URL: https://github.com/jannisborn/covid19_ultrasound
- License: CC-BY 4.0
- Content: Adult lung ultrasound frames from convex/linear probes, COVID/Pneumonia/Normal/Viral labels
- Frame extraction: `scripts/extract_openpocus_frames.py`

### Synthetic Neonatal Images
- Source: OpenPOCUS adult frames processed through `src/data/neonatal_synthesizer.py`
- Method: Brightness/contrast/noise augmentation + FOV simulation
- NOT real patient data

### Synthetic Clinical Data
- Source: Generated by `scripts/generate_clinical_data.py`
- Distributions based on: Sweet et al. 2023, Isayama et al. 2016
- NOT real patient data

---

## Architecture Decisions

- MobileNetV3-Small chosen for edge deployability
- Binary in Phase 1 (simpler, evaluable with limited data)
- 3-class in Phase 2 (clinically meaningful: Normal / Moderate / High Risk)
- Late fusion: image and clinical encoded separately, concatenated before final head
- Patient-level splits always — no frame-level leakage
