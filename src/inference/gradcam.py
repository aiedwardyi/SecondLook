"""Grad-CAM adapter for the 1-logit BCC model: attribution map plus probability."""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.bcc_model import gradcam_target_layer

# Heatmap opacity in the Grad-CAM overlay blend.
_HEATMAP_ALPHA = 0.4


def compute_gradcam(model, image_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """Grad-CAM map (HxW in [0, 1]) and BCC probability for one batched tile tensor."""
    targets = [ClassifierOutputTarget(0)]
    # enable_grad so the backward pass still runs if a caller wraps this in torch.no_grad().
    with torch.enable_grad(), GradCAM(model=model, target_layers=[gradcam_target_layer(model)]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        # Reuse the CAM forward pass; a second model() call would risk a device mismatch.
        prob = torch.sigmoid(cam.outputs).flatten()[0].item()
    return grayscale_cam[0], prob


def overlay_gradcam(rgb_image_01: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM map over an RGB tile in [0, 1]; returns a uint8 RGB overlay."""
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0.0, 1.0)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - _HEATMAP_ALPHA) * rgb_image_01 + _HEATMAP_ALPHA * heatmap
    return np.uint8(255 * blended)
