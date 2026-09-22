"""
Phase 1 Model — MobileNetV3-Small Binary Classifier
Binary: 0=Normal, 1=Abnormal LUS
"""

import torch
import torch.nn as nn
from torchvision import models


def create_phase1_model(pretrained: bool = True, freeze_backbone: bool = True, dropout: float = 0.3):
    """
    MobileNetV3-Small with custom binary classification head.
    ~1.5M params, ~0.06 GFLOPs — edge deployable.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(64, 2),  # binary
    )

    return model


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_phase1_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    model = create_phase1_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
