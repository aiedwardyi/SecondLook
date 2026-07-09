"""1-logit EfficientNet-B4 BCC model for binary BCC detection."""

import torch
import torch.nn as nn
import torchvision.models as models


class BccModel(nn.Module):
    """EfficientNet-B4 with a 1-logit head for binary BCC detection."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw logit, shape (batch, 1). BCEWithLogitsLoss applies sigmoid internally.
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def gradcam_target_layer(model) -> nn.Module:
    """Return the backbone layer Grad-CAM should hook for BCC attribution maps."""
    # CAM target class is ClassifierOutputTarget(0), the single positive logit.
    return model.backbone.features[-1]
