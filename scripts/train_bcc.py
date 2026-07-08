"""Two-phase BCC trainer: head-only warmup, then full fine-tune, on the Heidelberg H&E split."""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.table import Table
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.data.heidelberg_dataset import HeidelbergBccDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.metrics import compute_auroc, ppv_at_sens, sens_at_spec, spec_at_sens
from src.models.bcc_model import BccModel
from src.models.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)
console = Console()

_PHASE1_WARMUP_STEPS = 100
_GRAD_CLIP_MAX_NORM = 1.0


def _clean_nonfinite(record: dict) -> dict:
    """Map non-finite floats (NaN, inf, -inf) to None so each line is valid JSON for any parser."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in record.items()}


def _jsonl_append(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_clean_nonfinite(record), allow_nan=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed numpy and stdlib random per DataLoader worker; torch seeds its own RNG."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_pos_weight(train_dataset, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [train_dataset.get_imbalance_ratio()], dtype=torch.float32, device=device
    )


def evaluate(model, loader, device, use_amp, criterion) -> tuple[float, float, float, float, float]:
    """Val AUC, mean loss, and sens@spec=0.95 / spec@sens=0.95 / ppv@sens=0.95; AUC and operating points are NaN for a single-class split."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(x)
            # sklearn and the loss expect fp32; bf16 would coerce and add noise.
            losses.append(criterion(logits.float(), y).item())
            all_probs.extend(torch.sigmoid(logits.float()).cpu().numpy().flatten().tolist())
            all_labels.extend(y.cpu().numpy().flatten().astype(int).tolist())
    val_loss = float(np.mean(losses)) if losses else float("nan")
    val_auc = compute_auroc(all_labels, all_probs)
    val_sens = sens_at_spec(all_labels, all_probs, 0.95)
    val_spec = spec_at_sens(all_labels, all_probs, 0.95)
    val_ppv = ppv_at_sens(all_labels, all_probs, 0.95)
    return val_auc, val_loss, val_sens, val_spec, val_ppv


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp,
    warmup_scheduler=None,
    warmup_remaining=0,
) -> tuple[float, int]:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_GRAD_CLIP_MAX_NORM)
        optimizer.step()
        # Step the warmup only while it remains, so it cannot undo later plateau cuts.
        if warmup_scheduler is not None and warmup_remaining > 0:
            warmup_scheduler.step()
            warmup_remaining -= 1
        losses.append(loss.item())
    train_loss = float(np.mean(losses)) if losses else float("nan")
    return train_loss, warmup_remaining


def promote_best_checkpoint(model, output_dir: Path, device: torch.device, phase1_auc: float, phase2_auc: float) -> Path:
    """Write best.pth as the better of the two phases by val AUC, with checkpoint metadata."""
    best_phase1 = output_dir / "best_phase1.pth"
    best_phase2 = output_dir / "best_phase2.pth"
    if not best_phase1.exists() and not best_phase2.exists():
        raise RuntimeError(f"no checkpoint saved during training (output_dir={output_dir})")
    if best_phase2.exists() and phase2_auc >= phase1_auc:
        source = best_phase2
    else:
        source = best_phase1
        if best_phase2.exists():
            logger.warning("phase 2 did not improve on phase 1; promoting phase 1 as best.pth.")
    model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
    best = output_dir / "best.pth"
    save_checkpoint(model, best)
    return best


def _direction_color(value, prev, higher_is_better) -> str:
    """Value color if it beat prev in the wanted direction, worse-red if it regressed; baseline on first epoch or NaN."""
    if prev is None or math.isnan(value):
        return "#A9B1D6"
    improved = value > prev if higher_is_better else value < prev
    return "#7DCFFF" if improved else "#F7768E"


def _log_epoch(phase, epoch, train_loss, val_loss, val_auc, best, prev_val_auc, prev_val_loss, prev_train_loss) -> None:
    ac = _direction_color(val_auc, prev_val_auc, higher_is_better=True)
    lc = _direction_color(val_loss, prev_val_loss, higher_is_better=False)
    tc = _direction_color(train_loss, prev_train_loss, higher_is_better=False)
    gap = val_loss - train_loss
    prev_gap = None if (prev_val_loss is None or prev_train_loss is None) else prev_val_loss - prev_train_loss
    gc = _direction_color(abs(gap), None if prev_gap is None else abs(prev_gap), higher_is_better=False)
    best_tag = " [bold #4EC9B0]*best[/]" if best else ""
    logger.info(
        f"[#BB9AF7]{phase}[/] epoch [#7DCFFF]{epoch:02d}[/] | train [{tc}]{train_loss:.4f}[/] | "
        f"val [{lc}]{val_loss:.4f}[/] | gap [{gc}]{gap:+.4f}[/] | auc [{ac}]{val_auc:.4f}[/]{best_tag}"
    )


def _render_run_header(device, use_amp, n_train, n_val, n_test, epochs_phase1, epochs_phase2) -> None:
    """Log the run header: device/amp, tile counts, and both phase budgets."""
    logger.info("device [#E0AF68]%s[/] | amp [#E0AF68]%s[/]", device, use_amp)
    logger.info("train tiles [#E0AF68]%d[/] | val tiles [#E0AF68]%d[/] | test tiles [#E0AF68]%d[/]", n_train, n_val, n_test)
    logger.info("[#BB9AF7]phase 1:[/] head only for [#E0AF68]%d[/] epochs | [#BB9AF7]phase 2:[/] full fine-tune up to [#E0AF68]%d[/] epochs", epochs_phase1, epochs_phase2)


def _render_phase_transition(epochs_phase2, trainable_before, trainable_after) -> None:
    """Announce phase 2's budget and the just-completed backbone unfreeze."""
    logger.info("[#BB9AF7]phase 2:[/] full fine-tune for up to [#E0AF68]%d[/] epochs", epochs_phase2)
    logger.info("backbone unfrozen, scheduler reset | trainable params [#E0AF68]%d[/] -> [#E0AF68]%d[/]", trainable_before, trainable_after)


def _build_best_per_metric_table(epoch_history):
    """Build the end-of-training Best Per Metric Rich Table from epoch_history."""
    top_train_loss = min(epoch_history, key=lambda r: r["train_loss"])
    top_val_loss = min(epoch_history, key=lambda r: r["val_loss"])
    finite_auc_rows = [r for r in epoch_history if not math.isnan(r["val_auc"])]

    best_table = Table(title="Best Per Metric", box=box.ASCII)
    best_table.add_column("Metric", justify="left")
    best_table.add_column("Value", justify="right")
    best_table.add_column("Epoch", justify="right")

    best_table.add_row(
        "Train Loss",
        f"[#4EC9B0]{top_train_loss['train_loss']:.4f}[/]",
        f"{top_train_loss['epoch']:02d}",
    )
    best_table.add_row(
        "Val Loss",
        f"[#4EC9B0]{top_val_loss['val_loss']:.4f}[/]",
        f"{top_val_loss['epoch']:02d}",
    )
    if finite_auc_rows:
        top_val_auc = max(finite_auc_rows, key=lambda r: r["val_auc"])
        best_table.add_row(
            "Val AUC",
            f"[#4EC9B0]{top_val_auc['val_auc']:.4f}[/]",
            f"{top_val_auc['epoch']:02d}",
        )
    else:
        best_table.add_row("Val AUC", "[dim]n/a[/]", "[dim]-[/]")

    finite_sens_rows = [
        r for r in epoch_history if not math.isnan(r["val_sens_at_spec"])
    ]
    if finite_sens_rows:
        top_sens = max(finite_sens_rows, key=lambda r: r["val_sens_at_spec"])
        best_table.add_row(
            "Sens@Spec=0.95",
            f"[#4EC9B0]{top_sens['val_sens_at_spec']:.4f}[/]",
            f"{top_sens['epoch']:02d}",
        )
    else:
        best_table.add_row("Sens@Spec=0.95", "[dim]n/a[/]", "[dim]-[/]")

    finite_spec_rows = [
        r for r in epoch_history if not math.isnan(r["val_spec_at_sens"])
    ]
    if finite_spec_rows:
        top_spec = max(finite_spec_rows, key=lambda r: r["val_spec_at_sens"])
        best_table.add_row(
            "Spec@Sens=0.95",
            f"[#4EC9B0]{top_spec['val_spec_at_sens']:.4f}[/]",
            f"{top_spec['epoch']:02d}",
        )
    else:
        best_table.add_row("Spec@Sens=0.95", "[dim]n/a[/]", "[dim]-[/]")

    finite_ppv_rows = [r for r in epoch_history if not math.isnan(r["val_ppv_at_sens"])]
    if finite_ppv_rows:
        top_ppv = max(finite_ppv_rows, key=lambda r: r["val_ppv_at_sens"])
        best_table.add_row(
            "PPV@Sens=0.95",
            f"[#4EC9B0]{top_ppv['val_ppv_at_sens']:.4f}[/]",
            f"{top_ppv['epoch']:02d}",
        )
    else:
        best_table.add_row("PPV@Sens=0.95", "[dim]n/a[/]", "[dim]-[/]")

    return best_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase BCC trainer")
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs-phase1", type=int, default=12)
    parser.add_argument("--epochs-phase2", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _configure_logging(console) -> None:
    """Route logging through a Rich handler with markup on."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, highlighter=NullHighlighter(), rich_tracebacks=False, show_path=False, show_time=False, show_level=False)],
    )


def main() -> None:
    args = parse_args()
    _configure_logging(console)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    train_dataset = HeidelbergBccDataset(args.csv_path, args.data_root, "Train", train_transform)
    val_dataset = HeidelbergBccDataset(args.csv_path, args.data_root, "Validation", eval_transform)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "log.jsonl"
    log_path.write_text("", encoding="utf-8")
    _jsonl_append(log_path, {
        "event": "run_config",
        "csv": args.csv_path.name,
        "data_root": args.data_root.name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "early_stopping_patience": args.early_stopping_patience,
    })

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    pos_weight = build_pos_weight(train_dataset, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = BccModel(pretrained=args.pretrained).to(device)

    n_test = len(HeidelbergBccDataset(args.csv_path, args.data_root, "Test", eval_transform))
    _render_run_header(
        device, use_amp,
        len(train_dataset), len(val_dataset), n_test,
        args.epochs_phase1, args.epochs_phase2,
    )

    best_phase1 = args.output_dir / "best_phase1.pth"
    best_phase2 = args.output_dir / "best_phase2.pth"
    epoch = 0
    epoch_history: list[dict] = []
    prev_val_auc = None
    prev_val_loss = None
    prev_train_loss = None

    # Phase 1 trains the classifier head only. The backbone's weights are frozen
    # (requires_grad=False), but the model stays in train() mode so the backbone's
    # BatchNorm running statistics adapt to the H&E target domain during warmup.
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.1 + 0.9 * min((step + 1) / _PHASE1_WARMUP_STEPS, 1.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    best_val_auc_phase1 = float("-inf")
    warmup_remaining = _PHASE1_WARMUP_STEPS
    for _ in range(args.epochs_phase1):
        epoch += 1
        train_loss, warmup_remaining = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_amp,
            warmup_scheduler=warmup_scheduler,
            warmup_remaining=warmup_remaining,
        )
        val_auc, val_loss, val_sens, val_spec, val_ppv = evaluate(model, val_loader, device, use_amp, criterion)
        if not np.isnan(val_auc):
            scheduler.step(val_auc)
        saved = (not np.isnan(val_auc)) and val_auc > best_val_auc_phase1
        if saved:
            best_val_auc_phase1 = val_auc
            torch.save(model.state_dict(), best_phase1)
        _log_epoch("P1", epoch, train_loss, val_loss, val_auc, saved, prev_val_auc, prev_val_loss, prev_train_loss)
        if not np.isnan(val_auc):
            prev_val_auc = val_auc
        if not np.isnan(val_loss):
            prev_val_loss = val_loss
        if not np.isnan(train_loss):
            prev_train_loss = train_loss
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_sens_at_spec": val_sens,
                "val_spec_at_sens": val_spec,
                "val_ppv_at_sens": val_ppv,
                "gap": val_loss - train_loss,
                "saved": saved,
            }
        )
        _jsonl_append(log_path, {
            "event": "epoch", "phase": "phase1", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc,
            "val_sens_at_spec": val_sens, "val_spec_at_sens": val_spec, "val_ppv_at_sens": val_ppv,
            "gap": val_loss - train_loss, "saved": saved,
        })

    # Phase 2: unfreeze and fine-tune the whole network with a fresh optimizer.
    trainable_before = model.count_trainable_params()
    model.unfreeze_backbone()
    _render_phase_transition(args.epochs_phase2, trainable_before, model.count_trainable_params())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    best_val_auc_phase2 = float("-inf")
    epochs_no_improve = 0
    for _ in range(args.epochs_phase2):
        epoch += 1
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp
        )
        val_auc, val_loss, val_sens, val_spec, val_ppv = evaluate(model, val_loader, device, use_amp, criterion)
        if not np.isnan(val_auc):
            scheduler.step(val_auc)
        improved = (not np.isnan(val_auc)) and val_auc > best_val_auc_phase2
        if improved:
            best_val_auc_phase2 = val_auc
            torch.save(model.state_dict(), best_phase2)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        _log_epoch("P2", epoch, train_loss, val_loss, val_auc, improved, prev_val_auc, prev_val_loss, prev_train_loss)
        if not np.isnan(val_auc):
            prev_val_auc = val_auc
        if not np.isnan(val_loss):
            prev_val_loss = val_loss
        if not np.isnan(train_loss):
            prev_train_loss = train_loss
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_sens_at_spec": val_sens,
                "val_spec_at_sens": val_spec,
                "val_ppv_at_sens": val_ppv,
                "gap": val_loss - train_loss,
                "saved": improved,
            }
        )
        _jsonl_append(log_path, {
            "event": "epoch", "phase": "phase2", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc,
            "val_sens_at_spec": val_sens, "val_spec_at_sens": val_spec, "val_ppv_at_sens": val_ppv,
            "gap": val_loss - train_loss, "saved": improved,
        })
        if epochs_no_improve >= args.early_stopping_patience:
            logger.info("early stopping at epoch %d (%d without improvement)", epoch, epochs_no_improve)
            break

    best = promote_best_checkpoint(model, args.output_dir, device, best_val_auc_phase1, best_val_auc_phase2)
    logger.info("promoted best checkpoint -> %s", best)

    if epoch_history:
        console.print(_build_best_per_metric_table(epoch_history))


if __name__ == "__main__":
    main()
