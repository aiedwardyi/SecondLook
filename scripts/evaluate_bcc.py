"""Evaluate a trained BCC classifier on the test split: metrics and eval_summary.json."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.table import Table
from torch.amp import autocast
from torch.utils.data import DataLoader

from scripts.train_bcc import seed_worker
from src.config import HIGH_THRESH, LOW_THRESH
from src.data.heidelberg_dataset import HeidelbergBccDataset
from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import (
    compute_auroc,
    compute_three_tier,
    ppv_at_sens,
    sens_at_spec,
    spec_at_sens,
)
from src.models.bcc_model import BccModel
from src.models.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)
console = Console()


_TIER_ORDER = ["NEGATIVE", "UNCERTAIN", "POSITIVE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BCC binary classifier and write eval artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to experiments/<before|after>/. Default source of best.pth "
        "(overridable via --checkpoint) and default eval/ output (overridable "
        "via --output-dir).",
    )
    parser.add_argument(
        "--csv-path", type=Path, required=True, help="Path to tiles CSV"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root for relative file paths in CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint .pth path. When omitted, "
        "{run-dir}/best.pth is used (existing behavior).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit eval-artifacts output dir. When omitted, "
        "{run-dir}/eval is used (existing behavior).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap inference in torch.amp.autocast",
    )
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    return parser.parse_args()


def disable_amp_without_cuda(args: argparse.Namespace) -> None:
    """Disable AMP when the resolved device can't run bf16 autocast; no-op unless --use-amp was set."""
    if not args.use_amp:
        return
    if torch.device(args.device).type != "cuda":
        print(
            "WARNING: --use-amp requires CUDA; disabling AMP.",
            file=sys.stderr,
        )
        args.use_amp = False
        return
    if not torch.cuda.is_available():
        print(
            "WARNING: --use-amp requires CUDA, but CUDA is not available (PyTorch build lacks CUDA support, or no compatible driver/runtime found); disabling AMP.",
            file=sys.stderr,
        )
        args.use_amp = False
        return
    if not torch.cuda.is_bf16_supported():
        print(
            "WARNING: --use-amp uses bf16 autocast which requires a BF16-capable "
            "GPU (Ampere or newer); disabling AMP on this device.",
            file=sys.stderr,
        )
        args.use_amp = False


def _build_loader(
    csv_path: Path,
    data_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a HeidelbergBccDataset + DataLoader for the named split."""
    val_transform = get_eval_transforms()
    dataset = HeidelbergBccDataset(csv_path, data_root, split, val_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )


def _collect_predictions(
    model: BccModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[list[int], list[float]]:
    """Run model on loader; return (labels, probs) as Python lists."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(x)
            probs = torch.sigmoid(logits.float()).cpu().numpy().flatten().tolist()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy().flatten().astype(int).tolist())
    return all_labels, all_probs


def _build_summary_table(metrics: dict) -> Table:
    """Build the end-of-eval Rich summary Table (box.ASCII for cp1252 console safety)."""

    def _fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return "n/a" if not math.isfinite(v) else f"{v:.4f}"
        return str(v)

    table = Table(title="BCC Test Evaluation Summary", box=box.ASCII)
    table.add_column("Metric", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    table.add_row("Test AUC", _fmt(metrics["test_auc"]))
    table.add_row("Sens @ Spec=0.95", _fmt(metrics["sens_at_spec_095"]))
    table.add_row("Spec @ Sens=0.95", _fmt(metrics["spec_at_sens_095"]))
    table.add_row("PPV @ Sens=0.95", _fmt(metrics["ppv_at_sens_095"]))
    table.add_row("Test samples", _fmt(metrics["n_test_samples"]))

    counts = metrics["three_tier_counts"]
    for truth_label, bucket_key in (
        ("Negative", "negative_truth"),
        ("Positive", "positive_truth"),
    ):
        bucket = counts[bucket_key]
        cells = ", ".join(f"{tier}={bucket[tier]}" for tier in _TIER_ORDER)
        table.add_row(f"Verdict | truth={truth_label}", cells)

    return table


def _coerce_nonfinite(value):
    """Recursively convert non-finite floats to None for JSON serialization."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _coerce_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_nonfinite(v) for v in value]
    return value


def _write_eval_summary_json(metrics: dict, save_path: Path) -> None:
    """Write the metrics dict to JSON; non-finite floats are coerced to None upstream."""
    cleaned = _coerce_nonfinite(metrics)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Eval summary JSON saved -> %s", save_path)


def _evaluate(
    model: BccModel,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    eval_dir: Path,
    high_thresh: float,
    low_thresh: float,
    run_id: str,
    checkpoint_path: Path,
) -> dict:
    """Compute test-set metrics and write eval_summary.json; testable without main()."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_labels, test_probs = _collect_predictions(model, test_loader, device, use_amp)
    test_auc = compute_auroc(test_labels, test_probs)
    test_sens = sens_at_spec(test_labels, test_probs, target_spec=0.95)
    test_spec = spec_at_sens(test_labels, test_probs, target_sens=0.95)
    test_ppv = ppv_at_sens(test_labels, test_probs, target_sens=0.95)

    tier_labels = compute_three_tier(test_probs, high_thresh, low_thresh)

    three_tier_counts = {
        "negative_truth": {tier: 0 for tier in _TIER_ORDER},
        "positive_truth": {tier: 0 for tier in _TIER_ORDER},
    }
    for label_value, tier in zip(test_labels, tier_labels):
        bucket = "positive_truth" if int(label_value) == 1 else "negative_truth"
        three_tier_counts[bucket][tier] += 1

    metrics = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "n_val_samples": len(val_loader.dataset),
        "n_test_samples": len(test_labels),
        "test_auc": test_auc,
        "sens_at_spec_095": test_sens,
        "spec_at_sens_095": test_spec,
        "ppv_at_sens_095": test_ppv,
        "high_thresh": high_thresh,
        "low_thresh": low_thresh,
        "three_tier_counts": three_tier_counts,
    }
    _write_eval_summary_json(metrics, eval_dir / "eval_summary.json")
    return metrics


def main() -> None:
    args = parse_args()
    disable_amp_without_cuda(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    checkpoint_path = (
        args.checkpoint if args.checkpoint is not None else args.run_dir / "best.pth"
    )
    if not checkpoint_path.is_file():
        logger.error("Checkpoint not found or not a regular file: %s", checkpoint_path)
        sys.exit(1)

    eval_dir = args.output_dir if args.output_dir is not None else args.run_dir / "eval"
    if eval_dir.exists() and not eval_dir.is_dir():
        logger.error(
            "--output-dir points to an existing non-directory path: %s", eval_dir
        )
        sys.exit(1)

    val_loader = _build_loader(
        args.csv_path,
        args.data_root,
        "Validation",
        args.batch_size,
        args.num_workers,
    )
    test_loader = _build_loader(
        args.csv_path,
        args.data_root,
        "Test",
        args.batch_size,
        args.num_workers,
    )
    logger.info(
        "Val tiles: %d, test tiles: %d",
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = load_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    logger.info("Loaded model from %s", checkpoint_path)
    metrics = _evaluate(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        use_amp=args.use_amp,
        eval_dir=eval_dir,
        high_thresh=HIGH_THRESH,
        low_thresh=LOW_THRESH,
        run_id=args.run_dir.name,
        checkpoint_path=checkpoint_path,
    )

    console.print(_build_summary_table(metrics))


if __name__ == "__main__":
    main()
