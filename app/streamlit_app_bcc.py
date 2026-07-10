"""Streamlit demo: three-tier BCC verdict with live thresholds and a Grad-CAM evidence map."""

import hashlib
import io
import logging
import os
import pickle
import sys
import threading
from pathlib import Path

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

_UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_FILES = 25
_MAX_DECODE_PIXELS = 4096 * 4096
_CACHE_MAX = 16
_GRADCAM_LOCK = threading.Lock()
_GALLERY_DIR = _ROOT_PATH / "gallery"
_GALLERY_BUCKETS = ("positive", "unclear", "negative")
_GALLERY_BUCKET_LABELS = {
    "positive": "BCC",
    "unclear": "Borderline",
    "negative": "Non-BCC",
}

PAGE_STYLES = """
<style>
/* cap content width to prevent wide-window sprawl */
.block-container { padding-top: 3rem !important; padding-bottom: 0.5rem !important; max-width: 1100px !important; margin: 0 auto !important; }
.verdict-wrap {
    border-radius: 10px; padding: 12px 20px; margin: 8px 0;
    display: flex; align-items: center; gap: 14px;
}
.verdict-positive { background-color: #7b1010; border: 1px solid #c62828; }
.verdict-uncertain { background-color: #7a3800; border: 1px solid #ef6c00; }
.verdict-negative { background-color: #0d4a1a; border: 1px solid #2e7d32; }
.verdict-label { font-size: 1.35rem; font-weight: 800; letter-spacing: 1px; color: #ffffff !important; flex: 1; }
.verdict-prob { font-size: 1rem; font-weight: 500; color: rgba(255,255,255,0.85) !important; white-space: nowrap; }
.img-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px; opacity: 0.6;
    height: 1.2em; line-height: 1.2; white-space: nowrap; overflow: hidden;
}
.gallery-row-title {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; opacity: 0.72; margin: 1.1rem 0 0.35rem;
}
.gallery-score-wrap { margin-top: 0.55rem; }
.gallery-score-track {
    position: relative; height: 9px; border-radius: 999px; background: #d8dee6;
    border: 1px solid #c7ced8; overflow: visible; margin: 0.15rem 0 0.35rem;
}
.gallery-score-dot {
    position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 999px;
    background: #263238; border: 2px solid #ffffff; box-shadow: 0 0 0 1px rgba(0,0,0,0.18);
    transform: translate(-50%, -50%);
}
.gallery-threshold-line {
    position: absolute; top: -4px; bottom: -4px; width: 2px; background: #5f6b7a;
    transform: translateX(-50%);
}
.gallery-score-meta {
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
    font-size: 0.76rem; color: rgba(230,232,240,0.78);
}
.gallery-verdict {
    border-radius: 4px; padding: 0.12rem 0.38rem; color: #ffffff; font-weight: 800;
    font-size: 0.68rem;
}
.gallery-badge-positive { background-color: #7b1010; }
.gallery-badge-uncertain { background-color: #7a3800; }
.gallery-badge-negative { background-color: #0d4a1a; }
[data-testid="stTabs"] button { font-weight: 600; }
[data-testid="stSelectbox"] [data-baseweb="select"] { cursor: pointer !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { cursor: pointer !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stHeaderActionElements"] { display: none !important; }
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


def _pct(value: float) -> float:
    """Track coordinate for a 0 to 1 score or threshold."""
    return min(max(value, 0.0), 1.0) * 100.0


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


def _render_result(
    model, path: str, transform, high: float, low: float, image_bytes: bytes, rgb: np.ndarray
) -> None:
    """Verdict card plus the original/heatmap column pair for one decoded tile."""
    cam, prob = _infer(model, path, image_bytes, rgb, transform)
    _verdict_card(prob, high, low)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="img-label">Original tile</div>', unsafe_allow_html=True)
        st.image(rgb, width="stretch")
    with right:
        st.markdown('<div class="img-label">AI heatmap</div>', unsafe_allow_html=True)
        st.image(_overlay(rgb, cam), width="stretch")


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
        st.markdown(f'<div class="gallery-row-title">{bucket}</div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for col, tile_path in zip(cols, paths, strict=False):
            with col, st.container(border=True):
                st.image(str(tile_path), width="stretch")
                try:
                    image_bytes = tile_path.read_bytes()
                    rgb = _load_rgb(image_bytes)
                    _, score = _infer(model, path, image_bytes, rgb, transform)
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
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


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


def _examples(model, path: str, transform, high: float, low: float, model_id: str) -> None:
    """Examples tab; the clicked tile's result renders above the picker grid."""
    result_area = st.empty()
    _gallery_grid(model, path, transform, high, low, model_id)
    choice = st.session_state.get("gallery_choice")
    if choice and Path(choice).exists():
        with result_area.container():
            image_bytes = None
            try:
                image_bytes = Path(choice).read_bytes()
            except OSError:
                st.error("Failed to read tile from disk. The file may be missing or unreadable.")
            if image_bytes is not None:
                try:
                    rgb = _load_rgb(image_bytes)
                except ValueError as err:
                    st.error(str(err))
                except Exception:
                    st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
                else:
                    _render_result(model, path, transform, high, low, image_bytes, rgb)
    _scroll_to_top(st.session_state.get("pick_id", 0))


def _compare(transform, high: float, low: float) -> None:
    """Same tile under before/after weights with live Grad-CAM."""
    options = _model_options()
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
        ["before (correction off)", "after (correction on)"],
        index=1,
        horizontal=True,
        key="compare_side",
        label_visibility="collapsed",
    )
    st.caption(f"Weights: **{side}**")
    ckpt_path, expected_padding_mode, _ = options[side]
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
        st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
        return
    _render_result(model, ckpt_path, transform, high, low, image_bytes, rgb)


def _single(model, path: str, transform, high: float, low: float) -> None:
    upload = st.file_uploader("Tile", type=_UPLOAD_TYPES, key="single")
    if upload is None:
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
        st.error("Failed to read tile. The file may be corrupt or in an unsupported format.")
        return
    _render_result(model, path, transform, high, low, image_bytes, rgb)


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
            st.stop()
        oversized = [f for f in uploads if f.size > MAX_UPLOAD_BYTES]
        if oversized:
            names = ", ".join(f.name for f in oversized[:3])
            more = f" and {len(oversized) - 3} more" if len(oversized) > 3 else ""
            st.error(f"These files exceed the {MAX_UPLOAD_MB} MB limit: {names}{more}.")
            st.stop()
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
    if low >= high:
        st.error(f"NEGATIVE threshold ({low:.2f}) must be below POSITIVE threshold ({high:.2f}).")
        st.stop()
    st.sidebar.caption(f"Between {low:.2f} and {high:.2f} -> UNCERTAIN")
    st.sidebar.markdown("---")
    st.sidebar.caption("Research and educational demo. Not for clinical use.")

    ckpt_path, expected_padding_mode, model_id = options[label]
    if st.session_state.get("active_ckpt") != ckpt_path:
        st.session_state.active_ckpt = ckpt_path
        st.session_state.cache = {}
        st.session_state.batch = None
        st.session_state.gallery_choice = None
    if not Path(ckpt_path).exists():
        st.error(f"Model weights not found: {ckpt_path}")
        st.stop()
    try:
        model = _load_model(ckpt_path, expected_padding_mode)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, KeyError) as err:
        st.error(f"Could not load '{label}': {err}")
        st.stop()
    transform = get_eval_transforms()

    single_tab, batch_tab, examples_tab, compare_tab = st.tabs(
        ["Single Image Analysis", "Batch Analysis", "Examples", "Compare"]
    )
    with single_tab:
        _single(model, ckpt_path, transform, high, low)
    with batch_tab:
        _batch(model, ckpt_path, transform, high, low)
    with examples_tab:
        _examples(model, ckpt_path, transform, high, low, model_id)
    with compare_tab:
        _compare(transform, high, low)


if __name__ == "__main__":
    main()
