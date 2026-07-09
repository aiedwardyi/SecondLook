"""Write Grad-CAM evidence-map overlays for BCC tiles and print prob + three-tier label."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import compute_three_tier
from src.inference.gradcam import compute_gradcam, overlay_gradcam
from src.models.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_OVERLAY_SUFFIX = "_gradcam.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Grad-CAM overlays for BCC tiles and print prob + three-tier label.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained BCC checkpoint .pth",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A single tile image or a directory of tiles",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for overlay PNGs"
    )
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    return parser.parse_args()


def _collect_inputs(input_path: Path) -> list[Path]:
    """Tile paths for a single file or a directory tree (recursive), sorted; skips prior overlays."""
    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.suffix.lower() in _IMAGE_SUFFIXES
            and not p.name.endswith(_OVERLAY_SUFFIX)
        )
    return [input_path]


def _output_path(input_root: Path, output_dir: Path, tile: Path) -> Path:
    """Overlay path mirroring the tile's location under output_dir, so nested stems do not collide."""
    name = f"{tile.stem}{_OVERLAY_SUFFIX}"
    if input_root.is_dir():
        return output_dir / tile.relative_to(input_root).parent / name
    return output_dir / name


def _process_image(model, transform, path: Path, device) -> tuple[np.ndarray, float]:
    """Grad-CAM overlay (uint8 RGB) and BCC probability for one tile."""
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"))
    input_tensor = transform(image=arr)["image"].unsqueeze(0).to(device)
    cam, prob = compute_gradcam(model, input_tensor)
    orig_h, orig_w = arr.shape[:2]
    cam_full = (
        np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    display = arr.astype(np.float32) / 255.0
    return overlay_gradcam(display, cam_full), prob


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found or not a regular file: %s", args.checkpoint)
        sys.exit(1)

    paths = _collect_inputs(args.input)
    if not paths:
        logger.error("No tile images found at: %s", args.input)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    logger.info("Loaded model from %s", args.checkpoint)

    transform = get_eval_transforms()
    written = 0
    for path in paths:
        try:
            overlay, prob = _process_image(model, transform, path, device)
        except Exception:
            logger.exception("Failed to process %s; skipping", path)
            continue
        out_path = _output_path(args.input, args.output, path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        written += 1
        label = compute_three_tier([prob])[0]
        print(f"{path.name}  p={prob:.3f}  {label}")

    if written == 0:
        logger.error("No overlays written; all %d input(s) failed", len(paths))
        sys.exit(1)


if __name__ == "__main__":
    main()
