"""1-logit EfficientNet-B4 BCC model with a zeros/reflect padding toggle."""

import torch
import torch.nn as nn
import torchvision.models as models


def _apply_reflect_padding(module: nn.Module) -> None:
    """Switch padding convs in `module` to reflect padding to drop the zero-pad location shortcut."""
    for submodule in module.modules():
        if not isinstance(submodule, nn.Conv2d):
            continue
        padding = submodule.padding
        if isinstance(padding, tuple):
            has_pad = any(int(p) > 0 for p in padding)
        elif isinstance(padding, int):
            has_pad = padding > 0
        else:
            has_pad = False
        if has_pad:
            submodule.padding_mode = "reflect"


class BccModel(nn.Module):
    """EfficientNet-B4 with a 1-logit head for binary BCC detection.

    Pass padding_mode='reflect' for the after-correction model, 'zeros' for the before-correction model.
    """

    def __init__(self, pretrained: bool = False, padding_mode: str = "zeros") -> None:
        super().__init__()
        if padding_mode not in {"zeros", "reflect"}:
            raise ValueError(
                f"padding_mode must be 'zeros' or 'reflect', got {padding_mode!r}"
            )
        self.padding_mode = padding_mode
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        if padding_mode == "reflect":
            _apply_reflect_padding(self.backbone)
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
