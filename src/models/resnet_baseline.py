"""ResNet50 pre-addestrata usata come estrattore di feature dalle immagini."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBaseline(nn.Module):
    def __init__(self, out_dim: Optional[int] = None, pretrained: bool = True,
                 normalize_output: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Linear(self.feature_dim, out_dim) if out_dim else None
        self.normalize_output = normalize_output
        self.preprocess = weights.transforms() if weights is not None else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        if self.proj is not None:
            feats = self.proj(feats)
        if self.normalize_output:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
