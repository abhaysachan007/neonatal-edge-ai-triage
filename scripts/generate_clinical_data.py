"""
Synthetic Clinical Data Generator
===================================
Generates a synthetic neonatal clinical dataset based on
published RDS (Respiratory Distress Syndrome) cohort statistics.

References used for distributions:
  - Sweet et al. (2023) European Consensus Guidelines on RDS
  - Rojas-Reyes et al. Cochrane Review on surfactant therapy
  - Isayama et al. JAMA Pediatrics 2016 (SpO2/FiO2 targets)

Output CSV columns:
  patient_id, gestational_age_weeks, birth_weight_kg,
  spo2_percent, fio2_fraction, respiratory_rate,
  chest_retractions, grunting, nasal_flaring,
  lus_class (0=Normal, 1=Moderate, 2=High Risk),
  silverman_score (0-10, clinical severity)

Usage:
  python scripts/generate_clinical_data.py --n 500 --output data/synthetic/clinical_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_clinical_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Class distribution (roughly realistic) ──
    # Normal ~40%, Moderate ~35%, High Risk ~25%
    class_probs = [0.40, 0.35, 0.25]
    labels = rng.choice([0, 1, 2], size=n, p=class_probs)

    rows = []
    for i, label in enumerate(labels):

        # ── Gestational age (weeks) ──
        # Normal: term/near-term (36-40w)
        # Moderate: moderate preterm (30-35w)
        # High: very preterm (24-30w)
        if label == 0:
            ga = rng.normal(38.0, 1.5)
            ga = np.clip(ga, 34, 42)
        elif label == 1:
            ga = rng.normal(32.5, 2.0)
            ga = np.clip(ga, 28, 36)
        else:
            ga = rng.normal(27.0, 2.0)
            ga = np.clip(ga, 23, 31)

        # ── Birth weight (kg) — correlated with GA ──
        bw = 0.12 * ga - 1.75 + rng.normal(0, 0.18)
        bw = np.clip(bw, 0.45, 4.5)

        # ── SpO2 (%) ──
        if label == 0:
            spo2 = rng.normal(97, 1.5)
            spo2 = np.clip(spo2, 92, 100)
        elif label == 1:
            spo2 = rng.normal(93, 2.5)
            spo2 = np.clip(spo2, 85, 98)
        else:
            spo2 = rng.normal(86, 4.0)
            spo2 = np.clip(spo2, 70, 94)

        # ── FiO2 fraction ──
        if label == 0:
            fio2 = rng.normal(0.21, 0.02)
            fio2 = np.clip(fio2, 0.21, 0.28)
        elif label == 1:
            fio2 = rng.normal(0.35, 0.08)
            fio2 = np.clip(fio2, 0.21, 0.60)
        else:
            fio2 = rng.normal(0.65, 0.15)
            fio2 = np.clip(fio2, 0.40, 1.00)

        # ── Respiratory rate (breaths/min) ──
        if label == 0:
            rr = rng.normal(45, 8)
            rr = np.clip(rr, 30, 60)
        elif label == 1:
            rr = rng.normal(62, 10)
            rr = np.clip(rr, 45, 80)
        else:
            rr = rng.normal(78, 12)
            rr = np.clip(rr, 55, 100)

        # ── Binary clinical signs ──
        # Chest retractions
        if label == 0:
            retractions = int(rng.random() < 0.08)
        elif label == 1:
            retractions = int(rng.random() < 0.55)
        else:
            retractions = int(rng.random() < 0.92)

        # Grunting
        if label == 0:
            grunting = int(rng.random() < 0.05)
        elif label == 1:
            grunting = int(rng.random() < 0.40)
        else:
            grunting = int(rng.random() < 0.85)

        # Nasal flaring
        if label == 0:
            flaring = int(rng.random() < 0.06)
        elif label == 1:
            flaring = int(rng.random() < 0.45)
        else:
            flaring = int(rng.random() < 0.88)

        # ── Silverman-Anderson score (0-10) ──
        # Composed of: retractions + grunting + flaring + xiphoid retraction + nasal dilation
        # Simplified: derive from label + noise
        if label == 0:
            silverman = rng.integers(0, 3)
        elif label == 1:
            silverman = rng.integers(3, 6)
        else:
            silverman = rng.integers(5, 11)

        rows.append({
            "patient_id": f"SYN_{i:05d}",
            "gestational_age_weeks": round(float(ga), 1),
            "birth_weight_kg": round(float(bw), 3),
            "spo2_percent": round(float(spo2), 1),
            "fio2_fraction": round(float(fio2), 3),
            "respiratory_rate": int(round(rr)),
            "chest_retractions": retractions,
            "grunting": grunting,
            "nasal_flaring": flaring,
            "silverman_score": int(silverman),
            "lus_class": int(label),
            "lus_class_name": ["Normal", "Moderate Risk", "High Risk"][label],
            "data_source": "synthetic",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic neonatal clinical data")
    parser.add_argument("--n", type=int, default=500, help="Number of patients to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic/clinical_data.csv")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} synthetic patients (seed={args.seed})...")
    df = generate_clinical_data(n=args.n, seed=args.seed)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\nClass distribution:")
    print(df["lus_class_name"].value_counts())

    print("\nSummary stats:")
    print(df[["gestational_age_weeks", "birth_weight_kg", "spo2_percent", "fio2_fraction"]].describe().round(2))


if __name__ == "__main__":
    main()
