"""Read-only viewer: re-render a BCC run's saved training and test-eval tables from log.jsonl + eval/eval_summary.json."""

import argparse
import json
import math
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()

# Tokyo Night palette.
BLUE = "#7DCFFF"      # improving values, epoch number, header values
RED = "#F7768E"       # regressing values
BASELINE = "#A9B1D6"  # first epoch or NaN
PHASE = "#BB9AF7"     # phase labels
LR = "#E0AF68"        # learning rate
TEAL = "#4EC9B0"      # best/saved marker, Best Per Metric, eval summary

_NUMERIC = (
    "train_loss", "val_loss", "val_auc",
    "val_sens_at_spec", "val_spec_at_sens", "val_ppv_at_sens", "gap",
)


def _direction_color(value, prev, higher_is_better) -> str:
    """Value color if it beat prev in the wanted direction, worse-red if it regressed; baseline on first epoch or NaN."""
    if prev is None or math.isnan(prev) or math.isnan(value):
        return BASELINE
    improved = value > prev if higher_is_better else value < prev
    return BLUE if improved else RED


def _fmt(value) -> str:
    """Metric for display: None/non-finite -> n/a, finite float -> 4dp, else str."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def _cell(value, prev, higher_is_better) -> str:
    """A Training Summary metric cell: text via _fmt, color via _direction_color."""
    return f"[{_direction_color(value, prev, higher_is_better)}]{_fmt(value)}[/]"


def _load_log(log_path) -> tuple[dict | None, list[dict]]:
    """Return (run_config, epoch_rows) from log.jsonl; JSON null becomes NaN so isnan and color checks work."""
    run_config = None
    epoch_rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        if event == "run_config" and run_config is None:
            run_config = record
        elif event == "epoch":
            for key in _NUMERIC:
                if record.get(key) is None:
                    record[key] = float("nan")
            epoch_rows.append(record)
    epoch_rows.sort(key=lambda r: r.get("epoch", 0))
    return run_config, epoch_rows


def _build_header_lines(run_config) -> list[str]:
    """Run-config header: values blue, phase labels purple, lr amber."""
    return [
        f"seed [{BLUE}]{_fmt(run_config.get('seed'))}[/]",
        f"batch_size [{BLUE}]{_fmt(run_config.get('batch_size'))}[/] | "
        f"lr [{LR}]{_fmt(run_config.get('lr'))}[/]",
        f"[{PHASE}]phase 1:[/] head only for [{BLUE}]{_fmt(run_config.get('epochs_phase1'))}[/] epochs | "
        f"[{PHASE}]phase 2:[/] full fine-tune up to [{BLUE}]{_fmt(run_config.get('epochs_phase2'))}[/] epochs",
        f"early stopping patience [{BLUE}]{_fmt(run_config.get('early_stopping_patience'))}[/]",
    ]


def _build_training_table(epoch_rows) -> Table:
    """Per-epoch Training Summary with direction colors and a phase-2 section break."""
    table = Table(
        title="Training Summary",
        caption="Sens at spec>=0.95, Spec at sens>=0.95, PPV at sens>=0.95",
        box=box.ASCII,
    )
    for col in ("Epoch", "Train Loss", "Val Loss", "AUC", "Sens", "Spec", "PPV"):
        table.add_column(col, justify="right", no_wrap=True)
    table.add_column("Saved", justify="center", no_wrap=True)

    prev = {k: None for k in ("val_loss", "val_auc", "val_sens_at_spec", "val_spec_at_sens", "val_ppv_at_sens")}
    prev_phase = None
    for row in epoch_rows:
        if prev_phase == "phase1" and row.get("phase") == "phase2":
            table.add_section()
        prev_phase = row.get("phase")

        tc = BASELINE if prev["val_loss"] is None else BLUE  # train loss uses baseline or blue, never red
        table.add_row(
            f"[{BLUE}]{row['epoch']:02d}[/]",
            f"[{tc}]{_fmt(row['train_loss'])}[/]",
            _cell(row["val_loss"], prev["val_loss"], False),
            _cell(row["val_auc"], prev["val_auc"], True),
            _cell(row["val_sens_at_spec"], prev["val_sens_at_spec"], True),
            _cell(row["val_spec_at_sens"], prev["val_spec_at_sens"], True),
            _cell(row["val_ppv_at_sens"], prev["val_ppv_at_sens"], True),
            f"[bold {TEAL}]yes[/]" if row.get("saved") else "",
        )
        for key in prev:
            prev[key] = row[key]
    return table


def _build_best_per_metric_table(epoch_rows) -> Table:
    """Best value per metric, teal throughout (six rows)."""
    table = Table(title="Best Per Metric", box=box.ASCII)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_column("Epoch", justify="right")

    def loss_row(label, key) -> None:
        finite = [r for r in epoch_rows if not math.isnan(r[key])]
        if finite:
            top = min(finite, key=lambda r: r[key])
            table.add_row(label, f"[{TEAL}]{top[key]:.4f}[/]", f"{top['epoch']:02d}")
        else:
            table.add_row(label, "[dim]n/a[/]", "[dim]-[/]")

    def metric_row(label, key) -> None:
        finite = [r for r in epoch_rows if not math.isnan(r[key])]
        if finite:
            top = max(finite, key=lambda r: r[key])
            table.add_row(label, f"[{TEAL}]{top[key]:.4f}[/]", f"{top['epoch']:02d}")
        else:
            table.add_row(label, "[dim]n/a[/]", "[dim]-[/]")

    loss_row("Train Loss", "train_loss")
    loss_row("Val Loss", "val_loss")
    metric_row("Val AUC", "val_auc")
    metric_row("Sens@Spec=0.95", "val_sens_at_spec")
    metric_row("Spec@Sens=0.95", "val_spec_at_sens")
    metric_row("PPV@Sens=0.95", "val_ppv_at_sens")
    return table


def _build_eval_table(metrics) -> Table:
    """BCC Test Evaluation Summary from eval_summary.json, teal throughout."""
    table = Table(title="BCC Test Evaluation Summary", box=box.ASCII)
    table.add_column("Metric", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    for label, key in (
        ("Test AUC", "test_auc"),
        ("Sens @ Spec=0.95", "sens_at_spec_095"),
        ("Spec @ Sens=0.95", "spec_at_sens_095"),
        ("PPV @ Sens=0.95", "ppv_at_sens_095"),
        ("Test samples", "n_test_samples"),
    ):
        table.add_row(label, f"[{TEAL}]{_fmt(metrics.get(key))}[/]")
    counts = metrics.get("three_tier_counts")
    if isinstance(counts, dict):
        for truth_label, bucket_key in (("Negative", "negative_truth"), ("Positive", "positive_truth")):
            bucket = counts.get(bucket_key)
            if isinstance(bucket, dict):
                cells = ", ".join(f"{t}={_fmt(bucket.get(t))}" for t in ("NEGATIVE", "UNCERTAIN", "POSITIVE"))
                table.add_row(f"Verdict | truth={truth_label}", f"[{TEAL}]{cells}[/]")
    return table


def main(argv=None):
    """Render one run's saved training and test-eval tables (read-only)."""
    parser = argparse.ArgumentParser(
        description="Re-render a BCC run's saved training and test-eval tables (read-only)."
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Run directory with log.jsonl and eval/eval_summary.json.",
    )
    run_dir = parser.parse_args(argv).run_dir

    if not run_dir.is_dir():
        console.print(f"Run directory not found: {escape(str(run_dir))}")
        return 2

    console.print(f"Run: {escape(str(run_dir))}")

    log_path = run_dir / "log.jsonl"
    if not log_path.is_file():
        console.print(f"[dim]No log.jsonl in {escape(str(run_dir))}; skipping training tables.[/]")
    else:
        run_config, epoch_rows = _load_log(log_path)
        if run_config is not None:
            for line in _build_header_lines(run_config):
                console.print(line)
        if epoch_rows:
            console.print(_build_training_table(epoch_rows))
            console.print(_build_best_per_metric_table(epoch_rows))
        else:
            console.print(f"[dim]No epoch records in {escape(log_path.name)}; skipping training tables.[/]")

    eval_path = run_dir / "eval" / "eval_summary.json"
    if not eval_path.is_file():
        console.print(f"[dim]No eval/eval_summary.json in {escape(str(run_dir))}; skipping evaluation summary.[/]")
    else:
        try:
            metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(f"[dim]Could not parse {escape(eval_path.name)}: {escape(str(exc))}; skipping evaluation summary.[/]")
        else:
            if not isinstance(metrics, dict):
                console.print(f"[dim]Could not parse {escape(eval_path.name)}: not a JSON object; skipping evaluation summary.[/]")
            else:
                console.print(_build_eval_table(metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())
