"""Streamlit BCC app: three-tier verdict, Grad-CAM evidence, live trust card."""

import hashlib
import html
import io
import json
import logging
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

# streamlit puts app/ on sys.path, not the repo root, so make src importable.
_ROOT_PATH = Path(__file__).resolve().parents[1]
_ROOT = str(_ROOT_PATH)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import HIGH_THRESH, LOW_THRESH  # noqa: E402
from src.data.transforms import get_eval_transforms  # noqa: E402
from src.device import get_best_device  # noqa: E402
from src.evaluation.metrics import compute_three_tier  # noqa: E402
from src.inference.gradcam import compute_gradcam, overlay_gradcam  # noqa: E402
from src.models.checkpoint import load_checkpoint  # noqa: E402
from src.models.registry import resolve  # noqa: E402
from src.trust.auditor import (  # noqa: E402
    DEFER_EVIDENCE,
    DEFER_UNAVAILABLE,
    audit_tile,
    follow_up_attention,
    plain_reason_line,
)
from src.trust.heatmap_metrics import compute_heatmap_metrics  # noqa: E402

_UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_FILES = 25
_MAX_DECODE_PIXELS = 4096 * 4096
_CACHE_MAX = 16
_GRADCAM_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_AUDIT_LOG_PATH = _ROOT_PATH / "logs" / "attention_audits.jsonl"
_GALLERY_DIR = _ROOT_PATH / "gallery"
_GALLERY_BUCKETS = ("positive", "unclear", "negative")
_GALLERY_BUCKET_LABELS = {
    "positive": "Positive",
    "unclear": "Borderline",
    "negative": "Negative",
}
_TAB_LABELS = (
    "Single Image Analysis",
    "Batch Analysis",
    "Examples",
    "Compare",
)
_QUEUE_DEFER_DISPLAY = {
    DEFER_EVIDENCE: "Evidence was unclear",
    DEFER_UNAVAILABLE: "Attention check unavailable",
}
_STATUS_SUB = {
    "VERIFIED": "passed",
    "FLAGGED": "failed",
    "DEFER": "unresolved",
}

PAGE_STYLES = """
<style>
:root,
[data-theme="dark"] {
  --sl-bg: #0E1116;
  --sl-panel: #12151B;
  --sl-surface: #151923;
  --sl-surface-2: #1A1F29;
  --sl-line: rgba(255, 255, 255, 0.07);
  --sl-line-2: rgba(255, 255, 255, 0.14);
  --sl-text: #E9ECF2;
  --sl-text-2: #A7AFBD;
  --sl-text-3: #707A89;
  --sl-accent: #7CA5EC;
  --sl-chip: rgba(255, 255, 255, 0.05);
  --sl-hover: rgba(255, 255, 255, 0.045);
  --sl-ok: #4FBF8B;
  --sl-ok-bg: rgba(79, 191, 139, 0.09);
  --sl-ok-line: rgba(79, 191, 139, 0.35);
  --sl-warn: #D9A83F;
  --sl-warn-bg: rgba(217, 168, 63, 0.10);
  --sl-warn-line: rgba(217, 168, 63, 0.38);
  --sl-defer: #9AA8DC;
  --sl-defer-bg: rgba(154, 168, 220, 0.10);
  --sl-defer-line: rgba(154, 168, 220, 0.35);
  --sl-positive: #E0716C;
  --sl-positive-bg: rgba(224, 113, 108, 0.09);
  --sl-positive-line: rgba(224, 113, 108, 0.35);
}
[data-theme="light"] {
  --sl-bg: #F2F3F6;
  --sl-panel: #F8F9FB;
  --sl-surface: #FFFFFF;
  --sl-surface-2: #F5F6F9;
  --sl-line: rgba(18, 26, 40, 0.10);
  --sl-line-2: rgba(18, 26, 40, 0.19);
  --sl-text: #171E29;
  --sl-text-2: #49556A;
  --sl-text-3: #6E7A8F;
  --sl-accent: #2D5FC4;
  --sl-chip: rgba(18, 26, 40, 0.05);
  --sl-hover: rgba(18, 26, 40, 0.045);
  --sl-ok: #177A50;
  --sl-ok-bg: rgba(23, 122, 80, 0.07);
  --sl-ok-line: rgba(23, 122, 80, 0.32);
  --sl-warn: #96660F;
  --sl-warn-bg: rgba(150, 102, 15, 0.08);
  --sl-warn-line: rgba(150, 102, 15, 0.32);
  --sl-defer: #4C5CA8;
  --sl-defer-bg: rgba(76, 92, 168, 0.08);
  --sl-defer-line: rgba(76, 92, 168, 0.32);
  --sl-positive: #B0392F;
  --sl-positive-bg: rgba(176, 57, 47, 0.07);
  --sl-positive-line: rgba(176, 57, 47, 0.30);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--sl-bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
  padding: 2.125rem 3.5rem 9rem !important;
  max-width: 960px !important;
  margin: 0 auto !important;
}
h1 {
  margin-bottom: 0.9rem !important;
  font-size: 1.75rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
}
h2, h3 {
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
}
[data-testid="stSidebar"] {
  width: 296px !important;
  min-width: 296px !important;
  background: var(--sl-panel) !important;
  border-right: 1px solid var(--sl-line) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 0.4rem 0.35rem 1rem;
}
[data-testid="stSidebar"] h2 {
  font-size: 0.95rem !important;
  margin-bottom: 0.1rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--sl-line) !important;
  margin: 0.9rem 0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
  font-size: 0.78rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: var(--sl-text-3) !important;
  line-height: 1.5;
}
[data-testid="stSidebar"] details {
  border: 1px solid var(--sl-line) !important;
  border-radius: 0.75rem !important;
  background: var(--sl-surface) !important;
}
[data-testid="stSidebar"] details summary {
  font-size: 0.78rem;
  font-weight: 600;
}
[data-testid="stSidebar"] button {
  min-height: 2rem;
  border-color: var(--sl-line-2) !important;
}
[data-testid="stSidebar"] [data-testid="stCode"] {
  color: var(--sl-ok);
  background: var(--sl-ok-bg);
  border: 1px solid var(--sl-ok-line);
}
[data-testid="stFileUploaderDropzone"] {
  padding: 0.8rem 1rem !important;
  border-color: var(--sl-line-2) !important;
  border-radius: 0.75rem !important;
  background: var(--sl-surface) !important;
}
[data-testid="stFileUploaderDropzone"] button,
.stButton > button,
[data-testid="stPopover"] > button {
  border-color: var(--sl-line-2) !important;
  border-radius: 0.6rem !important;
  font-weight: 600 !important;
}
.stButton > button:hover,
[data-testid="stPopover"] > button:hover {
  border-color: var(--sl-accent) !important;
  background: var(--sl-hover) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0.125rem;
  border-bottom: 1px solid var(--sl-line);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  height: 2.7rem;
  padding: 0.55rem 0.85rem 0.7rem;
  color: var(--sl-text-3);
  font-size: 0.84rem;
  font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--sl-text) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  height: 2px;
  background: var(--sl-accent);
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  padding-top: 1.45rem;
}
[data-testid="stImage"] img {
  display: block;
  border: 1px solid var(--sl-line);
  border-radius: 0.75rem;
  background: #000000;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--sl-line) !important;
  border-radius: 0.8rem !important;
  background: var(--sl-surface) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.6rem !important;
}
[data-testid="stPopover"] {
  display: flex;
  justify-content: flex-end;
}
[data-testid="stPopover"] > button {
  min-height: 2rem;
  padding: 0.35rem 0.8rem;
  border-radius: 999px !important;
  color: var(--sl-text-2) !important;
  background: var(--sl-surface) !important;
}
div[data-baseweb="popover"] > div {
  border: 1px solid var(--sl-line-2) !important;
  border-radius: 1rem !important;
  background: var(--sl-surface) !important;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.32) !important;
}
div[data-baseweb="popover"] [data-testid="stCaptionContainer"] p {
  color: var(--sl-text-3) !important;
}
div[data-baseweb="popover"] input {
  background: var(--sl-bg) !important;
  border-color: var(--sl-line-2) !important;
}
[data-testid="stDialog"] > div {
  border: 1px solid var(--sl-line-2) !important;
  border-radius: 1.1rem !important;
  background: var(--sl-surface) !important;
}
.verdict-wrap {
    border-radius: 12px; padding: 11px 18px; margin: 4px 0 18px;
    display: flex; align-items: center; gap: 12px;
}
.verdict-positive { background: var(--sl-positive-bg); border: 1px solid var(--sl-positive-line); }
.verdict-uncertain { background: var(--sl-warn-bg); border: 1px solid var(--sl-warn-line); }
.verdict-negative { background: var(--sl-ok-bg); border: 1px solid var(--sl-ok-line); }
.verdict-positive .verdict-label, .verdict-positive .verdict-prob { color: var(--sl-positive) !important; }
.verdict-uncertain .verdict-label, .verdict-uncertain .verdict-prob { color: var(--sl-warn) !important; }
.verdict-negative .verdict-label, .verdict-negative .verdict-prob { color: var(--sl-ok) !important; }
.verdict-label { font-size: 0.88rem; font-weight: 700; letter-spacing: 0.07em; flex: 1; }
.verdict-prob { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-size: 0.82rem; font-weight: 600; white-space: nowrap; }
.img-label {
    font-size: 0.66rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 7px; color: var(--sl-text-3);
    height: 1.2em; line-height: 1.2; white-space: nowrap; overflow: hidden;
}
.gallery-row-title {
    font-size: 0.66rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--sl-text-3); margin: 0.2rem 0 0.6rem;
}
.gallery-score-wrap { margin-top: 0.5rem; }
.gallery-score-track {
    position: relative; height: 5px; border-radius: 999px; background: var(--sl-line-2);
    overflow: visible; margin: 0.25rem 0 0.45rem;
}
.gallery-score-dot {
    position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 999px;
    background: var(--sl-text); border: 2px solid var(--sl-surface); box-shadow: 0 0 0 1px var(--sl-line-2);
    transform: translate(-50%, -50%);
}
.gallery-threshold-line {
    position: absolute; top: -4px; bottom: -4px; width: 1px; background: var(--sl-text-3);
    transform: translateX(-50%);
}
.gallery-score-meta {
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.7rem; color: var(--sl-text-2);
}
.gallery-verdict {
    border-radius: 5px; padding: 0.12rem 0.42rem; font-family: system-ui, sans-serif;
    font-weight: 700; font-size: 0.6rem; letter-spacing: 0.06em;
}
.gallery-badge-positive { color: var(--sl-positive); background: var(--sl-positive-bg); border: 1px solid var(--sl-positive-line); }
.gallery-badge-uncertain { color: var(--sl-warn); background: var(--sl-warn-bg); border: 1px solid var(--sl-warn-line); }
.gallery-badge-negative { color: var(--sl-ok); background: var(--sl-ok-bg); border: 1px solid var(--sl-ok-line); }
[data-testid="stSelectbox"] [data-baseweb="select"] { cursor: pointer !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { cursor: pointer !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stHeaderActionElements"] { display: none !important; }

.trust {
  --trust-border: var(--sl-line);
  --trust-muted: var(--sl-text-3);
  --trust-surface: var(--sl-surface-2);
  border: 1px solid var(--trust-border);
  border-radius: 14px;
  background: var(--sl-surface);
  padding: 16px 20px 0;
  box-shadow: none;
  color: var(--sl-text);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 1.25rem 0 0.65rem;
  min-height: 9.4rem;
  overflow: hidden;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}
.trust.flag { border-color: var(--sl-warn-line); }
.trust.verify { border-color: var(--sl-ok-line); }
.trust.defer { border-color: var(--sl-defer-line); }
.trust.loading { border-color: var(--trust-border); }
.trust-head {
  display: flex; justify-content: space-between; align-items: center;
  gap: 10px; margin-bottom: 13px;
}
.brand {
  font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--trust-muted);
}
.live-badge {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
  padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--sl-line-2);
  color: var(--sl-text-3); background: var(--sl-chip);
  white-space: nowrap;
}
.live-badge.live, .trust.loading .live-badge {
  color: var(--sl-ok); background: var(--sl-ok-bg); border-color: var(--sl-ok-line);
}
.pill {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  padding: 4px 11px; border-radius: 999px; border: 1px solid;
}
.flag .pill { color: var(--sl-warn); background: var(--sl-warn-bg); border-color: var(--sl-warn-line); }
.verify .pill { color: var(--sl-ok); background: var(--sl-ok-bg); border-color: var(--sl-ok-line); }
.defer .pill { color: var(--sl-defer); background: var(--sl-defer-bg); border-color: var(--sl-defer-line); }
.sub {
  margin: 0 0 8px; font-size: 0.81rem; font-weight: 400;
  color: var(--sl-text-2);
}
.short { margin: 0; font-size: 0.84rem; line-height: 1.6; color: var(--sl-text); }
.short + .short { margin-top: 3px; color: var(--sl-text-2); }
.foot {
  margin: 15px -20px 0; padding: 10px 20px;
  border-top: 1px solid var(--trust-border); background: var(--sl-surface-2);
  font-size: 0.72rem; color: var(--trust-muted);
}
.route {
  margin-top: 13px; padding: 9px 13px; border-radius: 10px;
  background: var(--sl-chip);
  border: 1px solid var(--trust-border);
  font-size: 0.78rem; color: var(--sl-text-2);
}
.route strong { color: var(--sl-text); }
.trust-more {
  margin-top: 13px; font-size: 0.8rem; color: var(--trust-muted);
}
.trust-more > summary {
  list-style: none; cursor: pointer; user-select: none;
  display: flex; align-items: center; gap: 9px;
  font-size: 0.78rem; font-weight: 600; color: var(--sl-text-2);
  transition: opacity 160ms ease;
}
.trust-more > summary::-webkit-details-marker { display: none; }
.trust-more > summary::before {
  content: "";
  width: 0.45rem; height: 0.45rem;
  border-right: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor;
  transform: rotate(-45deg);
  transition: transform 180ms ease;
  margin-top: -1px;
}
.trust-more[open] > summary::before { transform: rotate(45deg); }
.trust-more-body {
  margin-top: 10px; padding-top: 2px;
  animation: trust-open 200ms ease;
}
.trust-evidence-note {
  margin: 0 0 11px; font-size: 0.8rem; line-height: 1.55; color: var(--sl-text-2);
}
.trust-metrics {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
  overflow: hidden; border: 1px solid var(--sl-line); border-radius: 11px;
  background: var(--sl-line);
}
.trust-metric { padding: 11px 14px; background: var(--sl-surface-2); }
.trust-metric-label {
  font-size: 0.62rem; font-weight: 600; letter-spacing: 0.07em;
  color: var(--sl-text-3); text-transform: uppercase;
}
.trust-metric-value {
  margin-top: 4px; font-size: 1.18rem; font-weight: 600;
  color: var(--sl-text); font-variant-numeric: tabular-nums;
}
.trust-metric-help { margin-top: 2px; font-size: 0.69rem; color: var(--sl-text-3); }
.trust-meta {
  margin-top: 12px; padding-top: 11px; border-top: 1px solid var(--sl-line);
  display: flex; flex-wrap: wrap; gap: 4px 13px;
  font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem; color: var(--sl-text-3);
}
.trust-strip {
  border: 1px solid var(--sl-line);
  border-radius: 13px;
  background: var(--sl-surface);
  padding: 14px 18px;
  margin: 1.25rem 0 1.1rem;
  color: var(--sl-text);
}
.trust-strip .brand {
  font-size: 0.8rem; font-weight: 600; letter-spacing: 0;
  text-transform: none; color: var(--sl-text);
}
.trust-strip p {
  margin: 4px 0 0; font-size: 0.82rem;
  color: var(--sl-text-2);
}
.trust.loading .pulse-line {
  height: 0.72rem; border-radius: 6px; margin-top: 8px;
  background: linear-gradient(
    90deg,
    var(--sl-chip) 0%,
    var(--sl-line-2) 50%,
    var(--sl-chip) 100%
  );
  background-size: 200% 100%;
  animation: trust-shimmer 1.4s ease-in-out infinite;
}
.trust.loading .pulse-line.w2 { width: 78%; }
.trust.loading .pulse-line.w3 { width: 62%; }
@keyframes trust-shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
@keyframes trust-open {
  from { opacity: 0; transform: translateY(-3px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  .trust, .trust-more > summary, .trust-more > summary::before { transition: none; }
  .trust.loading .pulse-line { animation: none; }
  .trust-more-body { animation: none; }
}
@media (max-width: 760px) {
  .block-container { padding: 1.5rem 1rem 7rem !important; }
  .trust-metrics { grid-template-columns: 1fr; }
  .trust-metric-help { display: none; }
}
</style>
"""


def _model_options() -> dict[str, tuple[str, str | None, str]]:
    """Ordered {label: (checkpoint_path, expected_padding_mode, gallery_id)} for shipped weights."""
    return {
        "before (correction off)": (
            os.environ.get("SECONDLOOK_BCC_BEFORE") or resolve("before").path,
            "zeros",
            "before",
        ),
        "after (correction on)": (
            os.environ.get("SECONDLOOK_BCC_AFTER") or resolve("after").path,
            "reflect",
            "after",
        ),
    }


def _model_plain_label(model_id: str) -> str:
    mid = str(model_id or "").strip().lower()
    if mid == "before":
        return "before model"
    if mid == "after":
        return "after model"
    return f"{mid} model" if mid else "model"


@st.cache_resource
def _load_model(path: str, expected_padding_mode: str | None):
    """Load and cache a BccModel keyed on its path and padding guard."""
    device = get_best_device()
    model = load_checkpoint(path, expected_padding_mode=expected_padding_mode, map_location=str(device))
    return model.to(device)


def _load_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to an (H, W, 3) uint8 RGB array."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        w, h = img.size
        if w * h > _MAX_DECODE_PIXELS:
            raise ValueError(f"Image is {w}x{h} px, over the {_MAX_DECODE_PIXELS:,}-pixel decode cap.")
        return np.array(img.convert("RGB"))


def _infer(model, path: str, image_bytes: bytes, rgb: np.ndarray, transform):
    """Cached (cam, prob) for one (checkpoint, tile); forward runs only on a cache miss."""
    key = (path, hashlib.sha256(image_bytes).hexdigest())
    cache = st.session_state.cache
    if key not in cache:
        tensor = transform(image=rgb)["image"].unsqueeze(0).to(get_best_device())
        with _GRADCAM_LOCK:
            cache[key] = compute_gradcam(model, tensor)
        if len(cache) > _CACHE_MAX:
            del cache[next(iter(cache))]
    return cache[key]


def _batch_prob(model, rgb: np.ndarray, transform) -> float:
    """Forward-only BCC score for one tile; no Grad-CAM, no cache write."""
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(get_best_device())
    with _GRADCAM_LOCK, torch.no_grad():
        return float(torch.sigmoid(model(tensor)).flatten()[0].item())


def _gallery_prob(model, tile_path: Path, transform, model_id: str) -> float:
    """Session-cached forward-only score for one gallery tile (threshold-independent)."""
    key = (str(tile_path), model_id)
    cache = st.session_state.setdefault("gallery_scores", {})
    if key not in cache:
        cache[key] = _batch_prob(model, _load_rgb(tile_path.read_bytes()), transform)
    return cache[key]


def _overlay(rgb_uint8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Grad-CAM overlay at the tile's native resolution (uint8 RGB)."""
    h, w = rgb_uint8.shape[:2]
    cam_full = (
        np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    return overlay_gradcam(rgb_uint8.astype(np.float32) / 255.0, cam_full)


def _overlay_png_bytes(rgb_uint8: np.ndarray, cam: np.ndarray) -> bytes:
    """In-memory PNG of the rendered Grad-CAM overlay."""
    arr = _overlay(rgb_uint8, cam)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _pct(value: float) -> float:
    """Track coordinate for a 0 to 1 score or threshold."""
    return min(max(value, 0.0), 1.0) * 100.0


def _metric_pct(value: float) -> str:
    return f"{int(round(float(value) * 100))}%"


def _score_track_html(score: float, high: float, low: float) -> str:
    """Score dot plus live threshold lines for one gallery tile."""
    verdict = compute_three_tier([score], high, low)[0]
    badge = {
        "POSITIVE": "gallery-badge-positive",
        "UNCERTAIN": "gallery-badge-uncertain",
        "NEGATIVE": "gallery-badge-negative",
    }[verdict]
    return f"""
    <div class="gallery-score-wrap">
        <div class="gallery-score-track" aria-label="model score track">
            <span class="gallery-threshold-line" style="left: {_pct(low):.3f}%"></span>
            <span class="gallery-threshold-line" style="left: {_pct(high):.3f}%"></span>
            <span class="gallery-score-dot" style="left: {_pct(score):.3f}%"></span>
        </div>
        <div class="gallery-score-meta">
            <span>model score {score:.4f}</span>
            <span class="gallery-verdict {badge}">{verdict}</span>
        </div>
    </div>
    """


def _verdict_card(prob: float, high: float, low: float) -> None:
    """Colored three-tier verdict card with the model score."""
    label = compute_three_tier([prob], high, low)[0]
    css = {
        "POSITIVE": "verdict-positive",
        "UNCERTAIN": "verdict-uncertain",
        "NEGATIVE": "verdict-negative",
    }[label]
    st.markdown(
        f"""
        <div class="verdict-wrap {css}">
            <span class="verdict-label">{label}</span>
            <span class="verdict-prob">Model score: {prob:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _verdict_cell_style(val: str) -> str:
    """Background/text color for one verdict cell in the batch table."""
    return {
        "POSITIVE": "background-color:#4a1010;color:#ef9a9a;font-weight:bold",
        "UNCERTAIN": "background-color:#4a2800;color:#ffcc80;font-weight:bold",
        "NEGATIVE": "background-color:#0d3318;color:#a5d6a7;font-weight:bold",
        "ERROR": "background-color:#2a2a2a;color:#aaaaaa;font-weight:bold",
    }.get(val, "")


def _canonical_metrics(metrics: dict) -> dict:
    return {
        "topk_mass": float(metrics["topk_mass"]),
        "corner_ratio": float(metrics["corner_ratio"]),
        "edge_ratio": float(metrics["edge_ratio"]),
        "topk_frac": float(metrics["topk_frac"]),
        "corner_frac": float(metrics["corner_frac"]),
        "edge_frac": float(metrics["edge_frac"]),
    }


def _evidence_hash(
    tile_bytes: bytes,
    model_id: str,
    ckpt_path: str,
    score: float,
    cam: np.ndarray,
    metrics: dict,
) -> str:
    """Stable key over tile bytes, model, score, CAM, and metrics for audit reuse."""
    h = hashlib.sha256()
    h.update(tile_bytes)
    h.update(model_id.encode("utf-8"))
    h.update(ckpt_path.encode("utf-8"))
    h.update(f"{float(score):.8f}".encode("ascii"))
    cam_arr = np.ascontiguousarray(cam, dtype=np.float32)
    h.update(hashlib.sha256(cam_arr.tobytes()).digest())
    canon = _canonical_metrics(metrics)
    h.update(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _append_audit_log(record: dict) -> None:
    try:
        with _LOG_LOCK:
            _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        logging.exception("audit log write failed")


def _safe_error(err: str | None) -> str | None:
    if not err:
        return None
    text = str(err)
    if "ANTHROPIC_API_KEY" in text or "api_key" in text.lower() or "sk-ant" in text:
        return "audit unavailable"
    return text[:200]


def _short_id(evidence_hash: str) -> str:
    return str(evidence_hash or "")[:8]


def _source_badge(source: str | None) -> str:
    if str(source or "") == "session_cache":
        return "Cached from this session"
    return "LIVE"


def _enqueue_review(payload: dict, audit: dict) -> str | None:
    """Add FLAGGED/DEFER evidence once; return queue transition text or None."""
    status = audit.get("status")
    if status not in {"FLAGGED", "DEFER"}:
        return None
    ehash = payload["evidence_hash"]
    queue: dict = st.session_state.review_queue
    if ehash in queue:
        return "In review queue"
    before = len(queue)
    reasons = [plain_reason_line(r) for r in (audit.get("reason_lines") or [])[:2]]
    queue[ehash] = {
        "evidence_hash": ehash,
        "model_id": payload["model_id"],
        "score": float(payload["score"]),
        "status": status,
        "defer_reason": audit.get("defer_reason"),
        "reason_lines": reasons,
        "rgb": payload["rgb"],
        "cam": payload["cam"],
        "metrics": _canonical_metrics(payload["metrics"]),
        "audit": dict(audit),
        "receipt": {
            "completed_at": audit.get("completed_at"),
            "latency_ms": audit.get("latency_ms"),
            "source": audit.get("source"),
        },
    }
    after = len(queue)
    return f"Added to review queue: {before} -> {after}"


def _get_or_run_audit(
    *,
    evidence_hash: str,
    score: float,
    tier: str,
    metrics: dict,
    rgb: np.ndarray,
    cam: np.ndarray,
    model_id: str,
) -> dict:
    """Return the cached Claude audit for this evidence, or make one live call."""
    cache: dict = st.session_state.audit_cache
    if evidence_hash in cache:
        cached = dict(cache[evidence_hash])
        cached["source"] = "session_cache"
        return cached

    png = _overlay_png_bytes(rgb, cam)
    t0 = time.perf_counter()
    result = audit_tile(
        score=score,
        verdict=tier,
        metrics=metrics,
        image_png=png,
        model_id=model_id,
    )
    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    completed_at = datetime.now(timezone.utc).isoformat()
    audit = result.to_dict()
    audit["completed_at"] = completed_at
    audit["latency_ms"] = latency_ms
    audit["source"] = "live"
    cache[evidence_hash] = dict(audit)
    _append_audit_log(
        {
            "ts": completed_at,
            "evidence_hash": evidence_hash,
            "model_id": model_id,
            "score": float(score),
            "tier": tier,
            "metrics": _canonical_metrics(metrics),
            "status": audit.get("status"),
            "defer_reason": audit.get("defer_reason"),
            "error": _safe_error(audit.get("error")),
            "reason_lines": list(audit.get("reason_lines") or []),
            "latency_ms": latency_ms,
            "source": "live",
        }
    )
    return audit


def _analyze(
    model,
    ckpt_path: str,
    transform,
    image_bytes: bytes,
    rgb: np.ndarray,
    model_id: str,
    high: float,
    low: float,
) -> dict:
    """Detector score + CAM only; never calls Claude. Records load-time tier."""
    cam, score = _infer(model, ckpt_path, image_bytes, rgb, transform)
    metrics = compute_heatmap_metrics(cam)
    ehash = _evidence_hash(image_bytes, model_id, ckpt_path, score, cam, metrics)
    load_tier = compute_three_tier([float(score)], high, low)[0]
    return {
        "score": float(score),
        "cam": cam,
        "rgb": rgb,
        "metrics": metrics,
        "evidence_hash": ehash,
        "model_id": model_id,
        "ckpt_path": ckpt_path,
        "load_tier": load_tier,
        "audit": None,
        "queue_note": None,
    }


def _run_attention_check(payload: dict, high: float, low: float) -> dict:
    """Claude audit is gated to POSITIVE detector tiers only."""
    score = float(payload["score"])
    tier = compute_three_tier([score], high, low)[0]
    if tier != "POSITIVE":
        return payload
    audit = _get_or_run_audit(
        evidence_hash=payload["evidence_hash"],
        score=score,
        tier=tier,
        metrics=payload["metrics"],
        rgb=payload["rgb"],
        cam=payload["cam"],
        model_id=payload["model_id"],
    )
    payload["audit"] = audit
    payload["queue_note"] = _enqueue_review(payload, audit)
    return payload


def _attach_cached_audit(payload: dict) -> dict:
    """Reuse session audit; badge is LIVE only for the just-finished call."""
    ehash = payload["evidence_hash"]
    live_flash = st.session_state.get("live_flash") == ehash
    if payload.get("audit") is not None:
        audit = dict(payload["audit"])
        if not live_flash and ehash in st.session_state.audit_cache:
            audit["source"] = "session_cache"
        payload["audit"] = audit
        return payload
    cached = st.session_state.audit_cache.get(ehash)
    if cached is None:
        return payload
    audit = dict(cached)
    audit["source"] = "live" if live_flash else "session_cache"
    payload["audit"] = audit
    payload["queue_note"] = _enqueue_review(payload, audit)
    return payload


def _loading_card_html() -> str:
    return dedent(
        """
        <div class="trust loading" id="trust-card-anchor">
          <div class="trust-head">
            <span class="brand">Claude attention check</span>
            <span class="live-badge">LIVE</span>
          </div>
          <p class="sub" style="color:var(--text-color);opacity:0.9">Checking model attention</p>
          <p class="short" style="opacity:0.72">Reviewing the score, attention map, and evidence metrics.</p>
          <div class="pulse-line"></div>
          <div class="pulse-line w2"></div>
          <div class="pulse-line w3"></div>
        </div>
        """
    ).strip()


def _neutral_strip_html(tier: str) -> str:
    if tier == "NEGATIVE":
        title = "Attention check not run"
        line = "This version audits positive calls only."
    else:
        title = "Human review already required"
        line = "No attention audit was run for this borderline call."
    return dedent(
        f"""
        <div class="trust-strip">
          <div class="brand">{html.escape(title)}</div>
          <p>{html.escape(line)}</p>
        </div>
        """
    ).strip()


def _trust_card_html(audit: dict, metrics: dict, queue_note: str | None, model_id: str, evidence_hash: str) -> str:
    status = str(audit.get("status") or "DEFER").upper()
    if status not in {"VERIFIED", "FLAGGED", "DEFER"}:
        status = "DEFER"
    css = {"VERIFIED": "verify", "FLAGGED": "flag", "DEFER": "defer"}[status]
    subtitle = _STATUS_SUB[status]
    reasons = [plain_reason_line(r) for r in (audit.get("reason_lines") or [])]
    main_reasons = reasons[:2]
    extra_reason = reasons[2] if len(reasons) > 2 else None
    reason_html = "".join(f'<p class="short">{html.escape(line)}</p>' for line in main_reasons)
    if not reason_html:
        reason_html = '<p class="short">No detail returned.</p>'
    m = _canonical_metrics(metrics)
    route_html = ""
    if status in {"FLAGGED", "DEFER"} and queue_note:
        route_html = f'<div class="route"><strong>Queue</strong> - {html.escape(queue_note)}</div>'
    badge = _source_badge(audit.get("source"))
    source_css = "live" if badge == "LIVE" else "cached"
    completed = audit.get("completed_at") or "-"
    latency = audit.get("latency_ms")
    latency_s = f"{int(latency)} ms" if latency is not None else "-"
    extra_html = (
        f'<p class="trust-evidence-note">{html.escape(extra_reason)}</p>' if extra_reason else ""
    )
    details = dedent(
        f"""
        {extra_html}
        <div class="trust-metrics">
          <div class="trust-metric">
            <div class="trust-metric-label">Focus concentration</div>
            <div class="trust-metric-value">{html.escape(_metric_pct(m['topk_mass']))}</div>
            <div class="trust-metric-help">how tightly heat gathers</div>
          </div>
          <div class="trust-metric">
            <div class="trust-metric-label">Corner heat</div>
            <div class="trust-metric-value">{html.escape(_metric_pct(m['corner_ratio']))}</div>
            <div class="trust-metric-help">heat on frame corners</div>
          </div>
          <div class="trust-metric">
            <div class="trust-metric-label">Edge heat</div>
            <div class="trust-metric-value">{html.escape(_metric_pct(m['edge_ratio']))}</div>
            <div class="trust-metric-help">heat along tile borders</div>
          </div>
        </div>
        <div class="trust-meta">
          <span>Model: {html.escape(_model_plain_label(model_id))}</span>
          <span>Evidence: {html.escape(_short_id(evidence_hash))}</span>
          <span>Completed: {html.escape(str(completed))}</span>
          <span>Latency: {html.escape(latency_s)}</span>
          <span>Source: {html.escape(badge)}</span>
        </div>
        """
    ).strip()
    return dedent(
        f"""
        <div class="trust {css}" id="trust-card-anchor">
          <div class="trust-head">
            <span class="brand">Claude attention check</span>
            <span class="live-badge {source_css}">{html.escape(badge)}</span>
          </div>
          <div class="trust-head" style="margin-bottom:6px">
            <span class="pill">{html.escape(status)}</span>
            <span class="sub" style="margin:0">{html.escape(subtitle)}</span>
          </div>
          {reason_html}
          {route_html}
          <details class="trust-more">
            <summary>View evidence</summary>
            <div class="trust-more-body">
              {details}
            </div>
          </details>
          <div class="foot">
            <div>Detector class unchanged - Final judgment remains with the reviewer</div>
          </div>
        </div>
        """
    ).strip()


def _scroll_to_selector(selector: str, nonce: str) -> None:
    sel = json.dumps(selector)
    components.html(
        f"""
        <script>
        // {html.escape(str(nonce))}
        const sel = {sel};
        const w = window.parent, d = w.document;
        const el = d.querySelector(sel);
        if (el) {{
          el.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
        }}
        </script>
        """,
        height=0,
    )


def _scroll_to_top(nonce: int) -> None:
    """Top-slot iframe; the nonce forces a reload so scroll fires only on a fresh pick."""
    components.html(
        f"""
        <script>
        // {nonce}
        const w = window.parent, d = w.document;
        const el = d.querySelector('[data-testid="stMain"]')
                || d.querySelector('section.main') || d.querySelector('.main');
        if (el) el.scrollTo({{top: 0, behavior: 'smooth'}});
        w.scrollTo({{top: 0, behavior: 'smooth'}});
        </script>
        """,
        height=0,
    )


def _maybe_scroll_result(result_key: str, evidence_hash: str) -> None:
    flag = f"scroll_result_{result_key}"
    if st.session_state.get(flag) == evidence_hash:
        _scroll_to_selector("#result-anchor", f"result-{evidence_hash[:8]}")
        st.session_state[flag] = None


def _maybe_scroll_trust(evidence_hash: str, source: str | None) -> None:
    if source != "live":
        return
    flag = "scroll_trust_once"
    if st.session_state.get(flag) == evidence_hash:
        _scroll_to_selector("#trust-card-anchor", f"trust-{evidence_hash[:8]}")
        st.session_state[flag] = None


def _render_followup_chat(payload: dict, high: float, low: float, result_key: str) -> None:
    audit = payload.get("audit")
    if not audit:
        return
    ehash = payload["evidence_hash"]
    history_key = f"chat_hist_{ehash}"
    hist: list = st.session_state.setdefault(history_key, [])
    with st.popover("Ask Claude", key=f"ask_{result_key}_{ehash[:8]}"):
        st.caption("Ask about this attention check")
        q = st.text_input(
            "Question",
            key=f"ask_q_{result_key}_{ehash[:8]}",
            max_chars=300,
            label_visibility="collapsed",
            placeholder="e.g. Why was this flagged?",
        )
        if st.button("Submit", key=f"ask_go_{result_key}_{ehash[:8]}"):
            question = (q or "").strip()
            if not question:
                st.warning("Enter a short question.")
            else:
                cache: dict = st.session_state.chat_cache
                norm = " ".join(question.split())
                ckey = (ehash, norm.lower())
                if ckey in cache:
                    answer = cache[ckey]
                else:
                    score = float(payload["score"])
                    tier = compute_three_tier([score], high, low)[0]
                    png = _overlay_png_bytes(payload["rgb"], payload["cam"])
                    answer, err = follow_up_attention(
                        score=score,
                        tier=tier,
                        metrics=payload["metrics"],
                        audit_status=str(audit.get("status") or ""),
                        reason_lines=list(audit.get("reason_lines") or []),
                        image_png=png,
                        question=question,
                    )
                    if err or not answer:
                        answer = "Claude follow-up unavailable"
                    else:
                        cache[ckey] = answer
                hist.append({"q": question, "a": answer})
                if len(hist) > 8:
                    del hist[:-8]
        for turn in hist[-6:]:
            st.markdown(f"**You:** {turn['q']}")
            st.caption(str(turn["a"]))


@st.dialog("Case review", width="large")
def _case_review_dialog(ehash: str) -> None:
    rec = (st.session_state.review_queue or {}).get(ehash)
    if not rec:
        st.warning("Case not found in the review queue.")
        return
    score = float(rec.get("score", 0.0))
    high = float(st.session_state.get("ui_high", HIGH_THRESH))
    low = float(st.session_state.get("ui_low", LOW_THRESH))
    verdict = compute_three_tier([score], high, low)[0]
    st.markdown(f"**{verdict}** · Model score: {score:.4f}")
    st.caption(f"{_model_plain_label(str(rec.get('model_id') or ''))}")
    rgb = rec.get("rgb")
    cam = rec.get("cam")
    if rgb is not None and cam is not None:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="img-label">Original tile</div>', unsafe_allow_html=True)
            st.image(rgb, width="stretch")
        with right:
            st.markdown(
                '<div class="img-label">Where the model looked</div>',
                unsafe_allow_html=True,
            )
            st.image(_overlay(rgb, cam), width="stretch")
    audit = rec.get("audit") or {}
    metrics = rec.get("metrics") or {}
    note = None
    if str(audit.get("status") or "") in {"FLAGGED", "DEFER"}:
        note = "In review queue"
    st.markdown(
        _trust_card_html(
            audit,
            metrics,
            note,
            str(rec.get("model_id") or ""),
            ehash,
        ),
        unsafe_allow_html=True,
    )


def _show_analyzed(payload: dict, high: float, low: float, result_key: str) -> None:
    score = float(payload["score"])
    tier = compute_three_tier([score], high, low)[0]
    st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)
    _maybe_scroll_result(result_key, payload["evidence_hash"])
    _verdict_card(score, high, low)
    rgb = payload["rgb"]
    cam = payload["cam"]
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="img-label">Original tile</div>', unsafe_allow_html=True)
        st.image(rgb, width="stretch")
    with right:
        st.markdown(
            '<div class="img-label">Where the model looked</div>',
            unsafe_allow_html=True,
        )
        st.image(_overlay(rgb, cam), width="stretch")

    if tier != "POSITIVE":
        st.markdown(_neutral_strip_html(tier), unsafe_allow_html=True)
        return

    payload = _attach_cached_audit(payload)
    audit = payload.get("audit")
    load_tier = payload.get("load_tier")
    trust_slot = st.empty()

    if audit is None and load_tier == "POSITIVE":
        trust_slot.markdown(_loading_card_html(), unsafe_allow_html=True)
        payload = _run_attention_check(payload, high, low)
        st.session_state[result_key] = payload
        audit = payload.get("audit")
        if audit and audit.get("source") == "live":
            st.session_state.live_flash = payload["evidence_hash"]
            st.session_state.scroll_trust_once = payload["evidence_hash"]
    elif audit is None:
        if st.button("Check attention", type="primary", key=f"check_attn_{result_key}"):
            trust_slot.markdown(_loading_card_html(), unsafe_allow_html=True)
            payload = _run_attention_check(payload, high, low)
            st.session_state[result_key] = payload
            audit = payload.get("audit")
            if audit and audit.get("source") == "live":
                st.session_state.live_flash = payload["evidence_hash"]
                st.session_state.scroll_trust_once = payload["evidence_hash"]
        else:
            return

    if audit is None:
        return

    # Consume one-shot LIVE flash after building display source.
    ehash = payload["evidence_hash"]
    if st.session_state.get("live_flash") == ehash:
        audit = dict(audit)
        audit["source"] = "live"
        st.session_state.live_flash = None
    elif ehash in st.session_state.audit_cache:
        audit = dict(audit)
        audit["source"] = "session_cache"

    trust_slot.markdown(
        _trust_card_html(
            audit,
            payload["metrics"],
            payload.get("queue_note"),
            payload["model_id"],
            ehash,
        ),
        unsafe_allow_html=True,
    )
    _maybe_scroll_trust(ehash, audit.get("source"))
    _render_followup_chat(payload, high, low, result_key)
    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)


def _gallery_tiles(model_id: str) -> list[tuple[str, Path]]:
    """Existing (bucket, path) for the 9 fixed slots under gallery/<model_id>/, in bucket order."""
    tiles = []
    for bucket in _GALLERY_BUCKETS:
        for i in (1, 2, 3):
            tile_path = _GALLERY_DIR / model_id / f"{bucket}_{i}.png"
            if tile_path.exists():
                tiles.append((bucket, tile_path))
    return tiles


def _gallery_grid(model, path: str, transform, high: float, low: float, model_id: str) -> None:
    """3x3 example-tile picker; a click stores the chosen tile path in gallery_choice."""
    tiles = _gallery_tiles(model_id)
    if not tiles:
        return
    for bucket in _GALLERY_BUCKETS:
        paths = [p for b, p in tiles if b == bucket]
        if not paths:
            continue
        heading = f"{_GALLERY_BUCKET_LABELS.get(bucket, bucket)} examples"
        st.markdown(f'<div class="gallery-row-title">{heading}</div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="small")
        for col, tile_path in zip(cols, paths, strict=False):
            with col, st.container(border=True):
                st.image(str(tile_path), width="stretch")
                try:
                    score = _gallery_prob(model, tile_path, transform, model_id)
                    st.markdown(
                        _score_track_html(score, high, low),
                        unsafe_allow_html=True,
                    )
                except Exception:
                    logging.exception("gallery score failed for %s", tile_path.name)
                if st.button(
                    "Analyze",
                    key=f"gallery_{model_id}_{tile_path.stem}",
                    width="stretch",
                ):
                    st.session_state.gallery_choice = str(tile_path)
                    st.session_state.pick_id = st.session_state.get("pick_id", 0) + 1
                    st.session_state.examples_result = None
                    st.session_state.examples_sig = None
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


def _examples(model, path: str, transform, high: float, low: float, model_id: str) -> None:
    """Examples tab; Analyze runs detector + CAM only for the selected gallery tile."""
    result_area = st.empty()
    _gallery_grid(model, path, transform, high, low, model_id)
    choice = st.session_state.get("gallery_choice")
    if choice and Path(choice).exists():
        with result_area.container():
            try:
                image_bytes = Path(choice).read_bytes()
                rgb = _load_rgb(image_bytes)
            except OSError:
                st.error("Failed to read tile from disk. The file may be missing or unreadable.")
            except ValueError as err:
                st.error(str(err))
            except Exception:
                logging.exception("examples tile decode failed for %s", choice)
                st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
            else:
                sig = (hashlib.sha256(image_bytes).hexdigest(), model_id, path, choice)
                if st.session_state.get("examples_sig") != sig:
                    st.session_state.examples_result = None
                if st.session_state.examples_result is None:
                    try:
                        with st.spinner("Analyzing..."):
                            st.session_state.examples_result = _analyze(
                                model, path, transform, image_bytes, rgb, model_id, high, low
                            )
                        st.session_state.examples_sig = sig
                    except Exception:
                        logging.exception("examples analyze failed")
                        st.error("Analysis failed. See server logs.")
                        st.session_state.examples_result = None
                if st.session_state.examples_result is not None:
                    _show_analyzed(st.session_state.examples_result, high, low, "examples_result")
    _scroll_to_top(st.session_state.get("pick_id", 0))


def _compare(transform, high: float, low: float) -> None:
    """Same tile under before/after weights; detector + CAM update on selection."""
    options = _model_options()
    model_labels = list(options)
    tiles = _gallery_tiles("after") or _gallery_tiles("before")
    if not tiles:
        st.info("No gallery tiles yet. Add curated tiles under gallery/before or gallery/after.")
        return
    labels = [
        f"{_GALLERY_BUCKET_LABELS.get(bucket, bucket)} {path.stem.rsplit('_', 1)[-1]}"
        for bucket, path in tiles
    ]
    paths = [path for _, path in tiles]
    pick = st.selectbox("Tile", range(len(labels)), format_func=lambda i: labels[i], key="compare_tile")
    side = st.radio(
        "Model",
        model_labels,
        index=1,
        horizontal=True,
        key="compare_side",
        label_visibility="collapsed",
    )
    st.caption(f"Weights: **{side}**")
    ckpt_path, expected_padding_mode, model_id = options[side]
    if not Path(ckpt_path).exists():
        st.error(f"Model weights not found: {ckpt_path}")
        return
    try:
        model = _load_model(ckpt_path, expected_padding_mode)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError) as err:
        st.error(f"Could not load '{side}': {err}")
        return
    tile_path = paths[pick]
    try:
        image_bytes = tile_path.read_bytes()
        rgb = _load_rgb(image_bytes)
    except OSError:
        st.error("Failed to read tile from disk.")
        return
    except ValueError as err:
        st.error(str(err))
        return
    except Exception:
        logging.exception("compare tile decode failed for %s", tile_path)
        st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
        return

    sig = (hashlib.sha256(image_bytes).hexdigest(), model_id, ckpt_path, str(tile_path), side)
    if st.session_state.get("compare_sig") != sig:
        st.session_state.compare_result = None
        st.session_state.compare_sig = sig
    if st.session_state.compare_result is None:
        try:
            with st.spinner("Analyzing..."):
                st.session_state.compare_result = _analyze(
                    model, ckpt_path, transform, image_bytes, rgb, model_id, high, low
                )
            st.session_state.compare_sig = sig
        except Exception:
            logging.exception("compare analyze failed")
            st.error("Analysis failed. See server logs.")
            st.session_state.compare_result = None
            return
    _show_analyzed(st.session_state.compare_result, high, low, "compare_result")


def _single(model, path: str, transform, high: float, low: float, model_id: str) -> None:
    upload = st.file_uploader("Tile", type=_UPLOAD_TYPES, key="single")
    if upload is None:
        st.session_state.single_result = None
        st.session_state.single_sig = None
        return
    if upload.size > MAX_UPLOAD_BYTES:
        st.error(
            f"File '{upload.name}' is {upload.size / (1024 * 1024):.1f} MB, "
            f"over the {MAX_UPLOAD_MB} MB limit."
        )
        return
    image_bytes = upload.getvalue()
    try:
        rgb = _load_rgb(image_bytes)
    except ValueError as err:
        st.error(str(err))
        return
    except Exception:
        logging.exception("single tile decode failed for %s", upload.name)
        st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
        return

    sig = (hashlib.sha256(image_bytes).hexdigest(), model_id, path)
    fresh = st.session_state.get("single_sig") != sig
    if fresh:
        st.session_state.single_result = None
        st.session_state.single_sig = sig
    if st.session_state.single_result is None:
        try:
            with st.spinner("Analyzing..."):
                st.session_state.single_result = _analyze(
                    model, path, transform, image_bytes, rgb, model_id, high, low
                )
            st.session_state.single_sig = sig
            st.session_state.scroll_result_single_result = st.session_state.single_result[
                "evidence_hash"
            ]
        except Exception:
            logging.exception("single analyze failed")
            st.error("Analysis failed. See server logs.")
            st.session_state.single_result = None
            return
    _show_analyzed(st.session_state.single_result, high, low, "single_result")


def _batch(model, path: str, transform, high: float, low: float) -> None:
    uploads = st.file_uploader(
        "Tiles", type=_UPLOAD_TYPES, accept_multiple_files=True, key="batch_upload"
    )
    signature = (path, tuple((f.name, f.size) for f in uploads or []))
    if st.session_state.get("batch_sig") != signature:
        st.session_state.batch = None
    if uploads:
        if len(uploads) > MAX_BATCH_FILES:
            st.error(f"Batch limited to {MAX_BATCH_FILES} tiles. You uploaded {len(uploads)}.")
            return
        oversized = [f for f in uploads if f.size > MAX_UPLOAD_BYTES]
        if oversized:
            names = ", ".join(f.name for f in oversized[:3])
            more = f" and {len(oversized) - 3} more" if len(oversized) > 3 else ""
            st.error(f"These files exceed the {MAX_UPLOAD_MB} MB limit: {names}{more}.")
            return
        st.info(f"{len(uploads)} tile(s) queued.")
        if st.button("Run Batch Analysis", type="primary", key="batch_run"):
            rows = []
            progress = st.progress(0)
            status = st.empty()
            for i, f in enumerate(uploads):
                status.text(f"Processing {f.name} ({i + 1}/{len(uploads)})...")
                try:
                    prob = _batch_prob(model, _load_rgb(f.getvalue()), transform)
                    rows.append({"name": f.name, "prob": prob})
                except Exception:
                    logging.exception("batch inference failed for %s", f.name)
                    rows.append({"name": f.name, "prob": None})
                progress.progress((i + 1) / len(uploads))
            status.text("Done.")
            st.session_state.batch = rows
            st.session_state.batch_sig = signature
            st.success(f"Processed {len(rows)} tiles.")
    rows = st.session_state.batch
    if rows:
        import pandas as pd

        table = []
        for r in rows:
            if r["prob"] is None:
                table.append({"Filename": r["name"], "Model score": "-", "Verdict": "ERROR"})
            else:
                label = compute_three_tier([r["prob"]], high, low)[0]
                table.append(
                    {"Filename": r["name"], "Model score": f"{r['prob']:.4f}", "Verdict": label}
                )
        df = pd.DataFrame(table)
        st.dataframe(
            df.style.map(_verdict_cell_style, subset=["Verdict"]),
            width="stretch",
            hide_index=True,
        )


def _queue_status_label(status: str, defer_reason: str | None) -> str:
    if status == "FLAGGED":
        return "Attention check failed"
    if status == "DEFER":
        return _QUEUE_DEFER_DISPLAY.get(str(defer_reason or ""), "Attention check unresolved")
    return str(status or "")


def _render_queue_sidebar() -> None:
    queue: dict = st.session_state.get("review_queue") or {}
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Review queue: {len(queue)}**")
    if not queue:
        st.sidebar.caption("No tiles queued.")
        return
    with st.sidebar.expander("Queue details", expanded=False):
        for ehash, rec in list(queue.items())[-8:]:
            status = str(rec.get("status") or "")
            label = _queue_status_label(status, rec.get("defer_reason"))
            with st.container(border=True):
                st.markdown(f"**{_model_plain_label(str(rec.get('model_id') or ''))}**")
                st.caption(f"Model score: {float(rec.get('score', 0)):.4f}")
                st.caption(f"{status} · {label}")
                for line in (rec.get("reason_lines") or [])[:2]:
                    st.caption(plain_reason_line(str(line)))
                if st.button("Open case", key=f"open_case_{ehash[:12]}"):
                    _case_review_dialog(ehash)


def main() -> None:
    st.set_page_config(
        page_title="SecondLook",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_STYLES, unsafe_allow_html=True)
    st.title("SecondLook")

    if "cache" not in st.session_state:
        st.session_state.cache = {}
    if "batch" not in st.session_state:
        st.session_state.batch = None
    if "batch_sig" not in st.session_state:
        st.session_state.batch_sig = None
    if "gallery_choice" not in st.session_state:
        st.session_state.gallery_choice = None
    if "audit_cache" not in st.session_state:
        st.session_state.audit_cache = {}
    if "chat_cache" not in st.session_state:
        st.session_state.chat_cache = {}
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = {}
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "single_sig" not in st.session_state:
        st.session_state.single_sig = None
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = None
    if "compare_sig" not in st.session_state:
        st.session_state.compare_sig = None
    if "examples_result" not in st.session_state:
        st.session_state.examples_result = None
    if "examples_sig" not in st.session_state:
        st.session_state.examples_sig = None

    options = _model_options()
    labels = list(options)
    st.sidebar.markdown("## BCC Inference")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model**")
    label = st.sidebar.radio(
        "Model", labels, index=labels.index("after (correction on)"), label_visibility="collapsed"
    )
    st.sidebar.markdown(f"**Device:** `{get_best_device().upper()}`")
    st.sidebar.markdown("**Detection Thresholds**")
    st.sidebar.caption("Drag the thresholds; tiles near a line flip live.")
    high = st.sidebar.slider("POSITIVE threshold", 0.50, 0.90, HIGH_THRESH, 0.05)
    low = st.sidebar.slider("NEGATIVE threshold", 0.10, 0.50, LOW_THRESH, 0.05)
    st.session_state.ui_high = high
    st.session_state.ui_low = low
    if low >= high:
        st.error(f"NEGATIVE threshold ({low:.2f}) must be below POSITIVE threshold ({high:.2f}).")
        st.stop()
    st.sidebar.caption(f"Between {low:.2f} and {high:.2f} -> UNCERTAIN")
    st.sidebar.markdown("---")
    st.sidebar.caption("Research and educational use. Not for clinical use.")

    ckpt_path, expected_padding_mode, model_id = options[label]
    if st.session_state.get("active_ckpt") != ckpt_path:
        st.session_state.active_ckpt = ckpt_path
        st.session_state.cache = {}
        st.session_state.batch = None
        st.session_state.gallery_choice = None
        st.session_state.single_result = None
        st.session_state.single_sig = None
        st.session_state.compare_result = None
        st.session_state.compare_sig = None
        st.session_state.examples_result = None
        st.session_state.examples_sig = None
    if not Path(ckpt_path).exists():
        st.error(f"Model weights not found: {ckpt_path}")
        st.stop()
    try:
        model = _load_model(ckpt_path, expected_padding_mode)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError) as err:
        st.error(f"Could not load '{label}': {err}")
        st.stop()
    transform = get_eval_transforms()

    tabs = st.tabs(_TAB_LABELS, key="main_tabs", on_change="rerun")
    single_tab, batch_tab, examples_tab, compare_tab = tabs

    if single_tab.open:
        with single_tab:
            _single(model, ckpt_path, transform, high, low, model_id)
    if batch_tab.open:
        with batch_tab:
            _batch(model, ckpt_path, transform, high, low)
    if examples_tab.open:
        with examples_tab:
            _examples(model, ckpt_path, transform, high, low, model_id)
    if compare_tab.open:
        with compare_tab:
            _compare(transform, high, low)

    _render_queue_sidebar()


if __name__ == "__main__":
    main()
