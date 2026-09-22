"""
Phase 2 Model — Multimodal Fusion (Image + Clinical)
MobileNetV3-Small image encoder + MLP clinical encoder → late fusion → 3-class triage

Classes: 0=Normal, 1=Moderate Risk, 2=High Risk
"""

import torch
import torch.nn as nn
from torchvision import models


class ClinicalEncoder(nn.Module):
    """Lightweight MLP for tabular clinical features."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """MobileNetV3-Small feature extractor (no classifier head)."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        if freeze:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Keep features + avgpool only, drop classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.out_dim = 576  # MobileNetV3-Small output channels

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


class NeonatalTriageModel(nn.Module):
    """
    Late fusion multimodal model.
    Image branch: MobileNetV3-Small → 576-dim
    Clinical branch: MLP → 32-dim
    Fusion: concat → 128-dim → 3-class softmax
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        clinical_input_dim: int = 8,
        clinical_hidden_dim: int = 32,
        fusion_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(pretrained=pretrained, freeze=freeze_backbone)
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_input_dim,
            hidden_dim=clinical_hidden_dim,
            output_dim=32,
        )

        fusion_input_dim = self.image_encoder.out_dim + 32

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)           # (B, 576)
        clin_feat = self.clinical_encoder(clinical)    # (B, 32)
        combined = torch.cat([img_feat, clin_feat], dim=1)  # (B, 608)
        return self.fusion_head(combined)              # (B, 3)


def create_phase2_model(**kwargs) -> NeonatalTriageModel:
    return NeonatalTriageModel(**kwargs)


if __name__ == "__main__":
    model = create_phase2_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    img = torch.randn(2, 3, 224, 224)
    clin = torch.randn(2, 8)
    out = model(img, clin)
    print(f"Output shape: {out.shape}")  # (2, 3)
