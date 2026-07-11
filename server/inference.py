"""Detector and Grad-CAM inference pipeline."""

import base64
import hashlib
import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from server.schemas import Metrics, ModelId, Tier
from src.config import HIGH_THRESH, LOW_THRESH
from src.data.transforms import get_eval_transforms
from src.device import get_best_device
from src.evaluation.metrics import compute_three_tier
from src.inference.gradcam import compute_gradcam, overlay_gradcam
from src.models.checkpoint import load_checkpoint
from src.models.registry import resolve
from src.trust.heatmap_metrics import compute_heatmap_metrics

_ROOT = Path(__file__).resolve().parents[1]
_CURATED_DIR = _ROOT / "gallery" / "before"
_MAX_DECODE_PIXELS = 4096 * 4096
_METRIC_KEYS = ("topk_mass", "corner_ratio", "edge_ratio")
_GRADCAM_LOCK = threading.Lock()

MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_TYPES = frozenset({"image/jpeg", "image/png", "image/tiff"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "TIFF"})

CANONICAL_TILES = {
    "positive-1": _CURATED_DIR / "positive_1.png",
    "positive-2": _CURATED_DIR / "positive_2.png",
    "positive-3": _CURATED_DIR / "positive_3.png",
    "borderline-1": _CURATED_DIR / "unclear_1.png",
    "borderline-2": _CURATED_DIR / "unclear_2.png",
    "borderline-3": _CURATED_DIR / "unclear_3.png",
    "negative-1": _CURATED_DIR / "negative_1.png",
    "negative-2": _CURATED_DIR / "negative_2.png",
    "negative-3": _CURATED_DIR / "negative_3.png",
}


@dataclass(frozen=True)
class InferenceResult:
    score: float
    tier: Tier
    metrics: Metrics
    overlay_png: bytes
    original_png: bytes
    evidence_hash: str


class InferencePipeline:
    def __init__(self, models: dict[ModelId, torch.nn.Module], device: torch.device) -> None:
        self.models = models
        self.device = device
        self.transform = get_eval_transforms()

    @classmethod
    def load(cls) -> "InferencePipeline":
        device = torch.device(get_best_device())
        models = {}
        for model_id in ("before", "after"):
            spec = resolve(model_id)
            env_name = f"SECONDLOOK_BCC_{model_id.upper()}"
            path = os.environ.get(env_name) or spec.path
            logging.info("Loading %s model from %s on %s", model_id, path, device)
            model = load_checkpoint(
                path,
                expected_padding_mode=spec.padding_mode,
                map_location=str(device),
            )
            models[model_id] = model.to(device=device, dtype=torch.float32)
            logging.info("Loaded %s model", model_id)
        return cls(models, device)

    def score(self, image_bytes: bytes, model_id: ModelId) -> float:
        """Forward-only probability; same transform/model as analyze, no Grad-CAM."""
        rgb = load_rgb(image_bytes)
        tensor = self.transform(image=rgb)["image"].unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        model = self.models[model_id]
        model.eval()
        with _GRADCAM_LOCK, torch.no_grad():
            return float(torch.sigmoid(model(tensor)).flatten()[0].item())

    def analyze(self, image_bytes: bytes, model_id: ModelId) -> InferenceResult:
        rgb = load_rgb(image_bytes)
        tensor = self.transform(image=rgb)["image"].unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        with _GRADCAM_LOCK:
            cam, score = compute_gradcam(self.models[model_id], tensor)
        raw_metrics = compute_heatmap_metrics(cam)
        metrics = Metrics(**{key: float(raw_metrics[key]) for key in _METRIC_KEYS})
        tier = compute_three_tier([float(score)], HIGH_THRESH, LOW_THRESH)[0]
        overlay_png = render_overlay_png(rgb, cam)
        original_png = encode_png(rgb)
        return InferenceResult(
            score=float(score),
            tier=tier,
            metrics=metrics,
            overlay_png=overlay_png,
            original_png=original_png,
            evidence_hash=build_evidence_hash(model_id, score, metrics, image_bytes),
        )


def canonical_tile_bytes(tile_id: str) -> bytes:
    try:
        path = CANONICAL_TILES[tile_id]
    except KeyError:
        valid = ", ".join(sorted(CANONICAL_TILES))
        raise ValueError(f"unknown tile_id {tile_id!r}; valid ids: {valid}") from None
    return path.read_bytes()


def validate_upload(image_bytes: bytes, content_type: str | None) -> None:
    if content_type not in ALLOWED_UPLOAD_TYPES:
        raise ValueError("unsupported image type")
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError("unsupported image format")
        image.verify()


def load_rgb(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if width * height > _MAX_DECODE_PIXELS:
            raise ValueError(
                f"Image is {width}x{height} px, over the {_MAX_DECODE_PIXELS:,}-pixel decode cap."
            )
        return np.array(image.convert("RGB"))


def render_overlay_png(rgb: np.ndarray, cam: np.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    cam_full = (
        np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (width, height), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    overlay = overlay_gradcam(rgb.astype(np.float32) / 255.0, cam_full)
    return encode_png(overlay)


def encode_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def png_data_url(image_png: bytes) -> str:
    encoded = base64.standard_b64encode(image_png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_evidence_hash(
    model_id: ModelId,
    score: float,
    metrics: Metrics,
    image_bytes: bytes,
) -> str:
    payload = {
        "model": model_id,
        "score": f"{float(score):.8f}",
        "metrics": {
            key: f"{float(getattr(metrics, key)):.8f}" for key in _METRIC_KEYS
        },
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
