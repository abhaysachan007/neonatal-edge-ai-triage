"""
Neonatal LUS Edge AI Triage — Gradio Demo

Tab 1 — Phase 1 Triage:   LUS image → Normal / Abnormal + confidence
Tab 2 — Phase 2 Full Triage: LUS image + 8 clinical inputs → Normal / Moderate / High Risk

Requires:
  pip install gradio
  python scripts/export_models.py   (creates TorchScript .pt files first)

Usage:
  python app.py
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────
P1_PT    = Path("models/phase1/phase1_traced.pt")
P2_PT    = Path("models/phase2/phase2_traced.pt")
P2_CKPT  = Path("models/phase2/checkpoints/phase2_best.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLINICAL_FEATURES = [
    "gestational_age_weeks",
    "birth_weight_kg",
    "spo2_percent",
    "fio2_fraction",
    "respiratory_rate",
    "chest_retractions",
    "grunting",
    "nasal_flaring",
]

# ── model loading ───────────────────────────────────────────────────────
def _load_models():
    models = {}

    if P1_PT.exists():
        models["phase1"] = torch.jit.load(str(P1_PT), map_location="cpu")
        models["phase1"].eval()
    else:
        print(f"[WARN] Phase 1 TorchScript not found: {P1_PT}")
        print("  Run: python scripts/export_models.py")

    if P2_PT.exists():
        models["phase2"] = torch.jit.load(str(P2_PT), map_location="cpu")
        models["phase2"].eval()
    else:
        print(f"[WARN] Phase 2 TorchScript not found: {P2_PT}")

    clin_mu  = None
    clin_std = None
    if P2_CKPT.exists():
        ckpt = torch.load(str(P2_CKPT), map_location="cpu", weights_only=False)
        mu_d  = ckpt.get("clinical_mu",  {})
        std_d = ckpt.get("clinical_std", {})
        clin_mu  = np.array([mu_d.get(f,  0.0) for f in CLINICAL_FEATURES], dtype=np.float32)
        clin_std = np.array([std_d.get(f, 1.0) for f in CLINICAL_FEATURES], dtype=np.float32)
        clin_std = np.where(clin_std == 0, 1.0, clin_std)

    return models, clin_mu, clin_std


MODELS, CLIN_MU, CLIN_STD = _load_models()


# ── image preprocessing ─────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _preprocess_image(pil_img):
    if pil_img is None:
        return None
    img = pil_img.convert("RGB")
    return _transform(img).unsqueeze(0)  # (1, 3, 224, 224)


# ── Tab 1: Phase 1 ─────────────────────────────────────────────────────
def predict_phase1(image):
    if image is None:
        return "Upload a LUS image first.", None
    if "phase1" not in MODELS:
        return "Phase 1 model not loaded. Run: python scripts/export_models.py", None

    img_t = _preprocess_image(image if isinstance(image, Image.Image) else Image.fromarray(image))
    with torch.no_grad():
        logits = MODELS["phase1"](img_t)
        probs  = torch.softmax(logits, dim=1)[0]

    normal_p   = float(probs[0])
    abnormal_p = float(probs[1])
    label = "Abnormal" if abnormal_p > 0.5 else "Normal"
    confidence = max(normal_p, abnormal_p)

    result = f"**{label}**  (confidence: {confidence*100:.1f}%)"
    bar    = {"Normal": round(normal_p, 4), "Abnormal": round(abnormal_p, 4)}
    return result, bar


# ── Tab 2: Phase 2 ─────────────────────────────────────────────────────
def predict_phase2(image,
                   gestational_age_weeks,
                   birth_weight_kg,
                   spo2_percent,
                   fio2_fraction,
                   respiratory_rate,
                   chest_retractions,
                   grunting,
                   nasal_flaring):
    if image is None:
        return "Upload a LUS image first.", None
    if "phase2" not in MODELS:
        return "Phase 2 model not loaded. Run: python scripts/export_models.py", None
    if CLIN_MU is None:
        return "Clinical normalization stats not found. Check phase2_best.pth.", None

    img_t = _preprocess_image(image if isinstance(image, Image.Image) else Image.fromarray(image))

    raw = np.array([
        gestational_age_weeks,
        birth_weight_kg,
        spo2_percent,
        fio2_fraction,
        respiratory_rate,
        float(chest_retractions),
        float(grunting),
        float(nasal_flaring),
    ], dtype=np.float32)
    norm = (raw - CLIN_MU) / CLIN_STD
    clin_t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)  # (1, 8)

    with torch.no_grad():
        logits = MODELS["phase2"](img_t, clin_t)
        probs  = torch.softmax(logits, dim=1)[0]

    class_names = ["Normal", "Moderate Risk", "High Risk"]
    pred_idx    = int(probs.argmax())
    label       = class_names[pred_idx]
    confidence  = float(probs[pred_idx])

    result = f"**{label}**  (confidence: {confidence*100:.1f}%)"
    bar    = {c: round(float(p), 4) for c, p in zip(class_names, probs)}
    return result, bar


# ── Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="Neonatal LUS Triage") as demo:
    gr.Markdown(
        "# Neonatal LUS Edge AI Triage\n"
        "> **Research prototype.** No real neonatal patient data used. "
        "Not a medical device."
    )

    with gr.Tab("Phase 1 — Binary Triage"):
        gr.Markdown("### Upload a lung ultrasound image → Normal / Abnormal")
        with gr.Row():
            img1   = gr.Image(label="LUS Image", type="pil")
            with gr.Column():
                out1_text = gr.Markdown(label="Prediction")
                out1_bar  = gr.Label(label="Confidence", num_top_classes=2)
        btn1 = gr.Button("Run Phase 1 Triage", variant="primary")
        btn1.click(predict_phase1, inputs=[img1], outputs=[out1_text, out1_bar])

    with gr.Tab("Phase 2 — Full Triage"):
        gr.Markdown(
            "### Upload LUS image + enter clinical vitals → "
            "Normal / Moderate Risk / High Risk"
        )
        with gr.Row():
            img2 = gr.Image(label="LUS Image", type="pil")
            with gr.Column():
                ga  = gr.Slider(22, 42, value=34, step=0.5,
                                label="Gestational Age (weeks)")
                bw  = gr.Slider(0.3, 5.0, value=2.0, step=0.05,
                                label="Birth Weight (kg)")
                sp  = gr.Slider(50, 100, value=92, step=1,
                                label="SpO2 (%)")
                fi  = gr.Slider(0.21, 1.0, value=0.30, step=0.01,
                                label="FiO2 (fraction)")
                rr  = gr.Slider(20, 120, value=60, step=1,
                                label="Respiratory Rate (breaths/min)")
                cr  = gr.Radio([0, 1], value=0, label="Chest Retractions (0=No, 1=Yes)")
                gr_ = gr.Radio([0, 1], value=0, label="Grunting (0=No, 1=Yes)")
                nf  = gr.Radio([0, 1], value=0, label="Nasal Flaring (0=No, 1=Yes)")
        with gr.Row():
            out2_text = gr.Markdown(label="Prediction")
            out2_bar  = gr.Label(label="Class Probabilities", num_top_classes=3)
        btn2 = gr.Button("Run Full Triage", variant="primary")
        btn2.click(
            predict_phase2,
            inputs=[img2, ga, bw, sp, fi, rr, cr, gr_, nf],
            outputs=[out2_text, out2_bar],
        )

    gr.Markdown(
        "---\n"
        "**Ethical Note:** This system is a research prototype built on synthetic data. "
        "Results cannot be extrapolated to clinical performance without validation on "
        "real neonatal data. Not a medical device."
    )


if __name__ == "__main__":
    demo.launch(share=False)
