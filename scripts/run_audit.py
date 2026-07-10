"""Run attention metrics and optional Claude audit on a Grad-CAM map."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.trust.auditor import audit_tile
from src.trust.heatmap_metrics import compute_heatmap_metrics


def _load_cam(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attention metrics and Claude audit")
    p.add_argument(
        "--cam",
        required=True,
        help="Grayscale Grad-CAM map preferred; RGB overlay is approximate only",
    )
    p.add_argument("--score", type=float, required=True, help="Detector score in [0, 1]")
    p.add_argument("--verdict", required=True, help="POSITIVE / NEGATIVE / UNCERTAIN")
    p.add_argument("--model-id", default="unknown", help="before / after / other label")
    p.add_argument("--image", default=None, help="Optional overlay path for Claude vision")
    p.add_argument("--metrics-only", action="store_true", help="Skip Claude API call")
    p.add_argument("--out", default=None, help="Append JSONL path (default stdout only)")
    args = p.parse_args(argv)

    cam_path = Path(args.cam)
    cam = _load_cam(cam_path)
    metrics = compute_heatmap_metrics(cam)

    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cam": str(cam_path),
        "score": args.score,
        "verdict": args.verdict,
        "model_id": args.model_id,
        "metrics": metrics,
    }

    if args.metrics_only:
        record["audit"] = None
    else:
        result = audit_tile(
            score=args.score,
            verdict=args.verdict,
            metrics=metrics,
            image_path=args.image or str(cam_path),
            model_id=args.model_id,
        )
        record["audit"] = result.to_dict()

    line = json.dumps(record, sort_keys=True)
    print(line)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
