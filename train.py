#!/usr/bin/env python3
"""
SEN2NEON training driver.

Trains one of the three models on 12-band 4x multispectral
super-resolution patches produced by the preprocessing pipeline
(create_splits.py -> create_patches.py -> normalize.py -> verify_dataset.py).

Expects, under --data-root:
    patches/all_patches.csv     (written by preprocessing/create_patches.py)

Produces, under --output-dir:
    checkpoints/last.pt
    checkpoints/best.pt
    history.csv
    config.used.yaml

Checkpoints are written in the format evaluation/evaluate.py and
evaluation/visualize.py already expect:
    {
        "model_state_dict": ...,
        "model_config": {...},      # kwargs needed to reconstruct the model
        ...
    }

Example:
    python train.py --config config.yaml --model hybrid --data-root SEN2NEON_5GB

    # quick smoke test on a handful of patches, no GPU required
    python train.py --config config.yaml --model cnn --epochs 1 \
        --max-train-samples 8 --max-val-samples 4 --num-workers 0

    # resume
    python train.py --config config.yaml --resume runs/sen2neon/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for train.py. Install it with: "
        "pip install pyyaml"
    ) from exc


# ============================================================
# PROJECT LAYOUT
#
#   <project root>/
#   |-- train.py            (this file)
#   |-- losses.py
#   |-- config.yaml
#   |-- dataset.py
#   |-- models/
#   |   |-- baseline_cnn.py
#   |   |-- transformer.py
#   |   `-- proposed_model.py
#   `-- evaluation/
#       |-- metrics.py
#       |-- evaluate.py
#       `-- visualize.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import create_dataloader  # noqa: E402
from losses import build_loss  # noqa: E402
from evaluation.metrics import calculate_all_metrics  # noqa: E402
from models.baseline_cnn import BaselineCNN  # noqa: E402
from models.transformer import TransformerSR  # noqa: E402
from models.proposed_model import HybridCNNTransformerSR  # noqa: E402


SCALE = 4
METRIC_KEYS = (
    "mae", "rmse", "psnr_db", "ssim",
    "sam_degrees", "ergas", "spectral_correlation",
)


# ============================================================
# CONFIG HANDLING
# ============================================================

def load_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a mapping.")
    return config


def deep_get(config: dict, dotted_key: str, default=None):
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def deep_set(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """CLI flags win over the YAML file; both are optional."""
    config = copy.deepcopy(config)

    overrides = {
        "data.data_root": str(args.data_root) if args.data_root is not None else None,
        "data.patch_index": str(args.patch_index) if args.patch_index is not None else None,
        "data.num_workers": args.num_workers,
        "data.max_train_samples": args.max_train_samples,
        "data.max_val_samples": args.max_val_samples,
        "model.name": args.model,
        "training.epochs": args.epochs,
        "training.batch_size": args.batch_size,
        "training.lr": args.lr,
        "training.seed": args.seed,
        "output.output_dir": str(args.output_dir) if args.output_dir is not None else None,
    }

    for key, value in overrides.items():
        if value is not None:
            deep_set(config, key, value)

    if args.no_amp:
        deep_set(config, "training.amp", False)

    return config


DEFAULTS = {
    "data": {
        "data_root": "SEN2NEON_5GB",
        "patch_index": None,
        "num_workers": 4,
        "return_masks": True,
        "max_train_samples": None,
        "max_val_samples": None,
    },
    "model": {
        "name": "hybrid",
        "scale": 4,
        "cnn": {"features": 64, "num_blocks": 8, "residual_scale": 0.1},
        "transformer": {
            "embed_dim": 96, "depth": 4, "num_heads": 4,
            "window_size": 8, "dropout": 0.0,
        },
        "hybrid": {
            "features": 64, "transformer_dim": 64, "cnn_blocks": 6,
            "transformer_blocks": 4, "heads": 4, "window_size": 8,
        },
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 3,
        "grad_clip": 1.0,
        "amp": True,
        "seed": 42,
        "val_interval": 1,
        "early_stopping_patience": 15,
        "early_stopping_metric": "psnr_db",
        "higher_is_better": True,
        "log_interval": 20,
    },
    "loss": {
        "l1_weight": 1.0,
        "charbonnier": True,
        "charbonnier_eps": 1e-3,
        "sam_weight": 0.1,
        "ssim_weight": 0.1,
        "ssim_window": 11,
    },
    "output": {
        "output_dir": "runs/sen2neon",
        "save_last": True,
        "save_best": True,
    },
}


def merge_defaults(config: dict) -> dict:
    def merge(base: dict, override: dict) -> dict:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(DEFAULTS, config)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL FACTORY (mirrors evaluation/evaluate.py::build_model)
# ============================================================

def build_model_and_config(
    model_name: str,
    model_cfg: dict,
    scale: int,
) -> Tuple[nn.Module, Dict[str, object]]:

    if model_name == "cnn":
        cnn_cfg = model_cfg.get("cnn", {})
        config = {
            "in_channels": 12,
            "out_channels": 12,
            "features": int(cnn_cfg.get("features", 64)),
            "num_blocks": int(cnn_cfg.get("num_blocks", 8)),
            "scale": int(scale),
            "residual_scale": float(cnn_cfg.get("residual_scale", 0.1)),
        }
        model = BaselineCNN(**config)

    elif model_name == "transformer":
        t_cfg = model_cfg.get("transformer", {})
        config = {
            "in_channels": 12,
            "out_channels": 12,
            "embed_dim": int(t_cfg.get("embed_dim", 96)),
            "depth": int(t_cfg.get("depth", 4)),
            "num_heads": int(t_cfg.get("num_heads", 4)),
            "window_size": int(t_cfg.get("window_size", 8)),
            "scale": int(scale),
            "dropout": float(t_cfg.get("dropout", 0.0)),
        }
        model = TransformerSR(**config)

    elif model_name == "hybrid":
        h_cfg = model_cfg.get("hybrid", {})
        config = {
            "in_channels": 12,
            "out_channels": 12,
            "features": int(h_cfg.get("features", 64)),
            "transformer_dim": int(h_cfg.get("transformer_dim", 64)),
            "cnn_blocks": int(h_cfg.get("cnn_blocks", 6)),
            "transformer_blocks": int(h_cfg.get("transformer_blocks", 4)),
            "heads": int(h_cfg.get("heads", 4)),
            "window_size": int(h_cfg.get("window_size", 8)),
            "scale": int(scale),
        }
        model = HybridCNNTransformerSR(**config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, config


# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================

def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 2e-4))
    wd = float(train_cfg.get("weight_decay", 1e-5))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
):
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 100))
    warmup = int(train_cfg.get("warmup_epochs", 0))

    if name == "none":
        return None

    if name == "plateau":
        mode = "max" if train_cfg.get("higher_is_better", True) else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=5,
        )

    if name == "cosine":
        total = max(epochs - warmup, 1)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / float(total)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {name}")


def scheduler_step(scheduler, metric_value: Optional[float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric_value is not None and math.isfinite(metric_value):
            scheduler.step(metric_value)
    else:
        scheduler.step()


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    model_name: str,
    model_config: dict,
    best_metric: Optional[float],
    full_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "global_step": global_step,
        "model_name": model_name,
        "model_config": model_config,
        "best_metric": best_metric,
        "config": full_config,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint_for_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, int, Optional[float]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    state = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"WARNING: resume is missing {len(missing)} model keys.")
    if unexpected:
        print(f"WARNING: resume ignored {len(unexpected)} unexpected model keys.")

    if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError) as exc:
            print(f"WARNING: could not restore optimizer state: {exc}")

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError) as exc:
            print(f"WARNING: could not restore scheduler state: {exc}")

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = checkpoint.get("best_metric", None)

    return start_epoch, global_step, best_metric


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def move_batch(batch: dict, device: torch.device) -> dict:
    out = {"lr": batch["lr"].to(device, non_blocking=True)}
    out["hr"] = batch["hr"].to(device, non_blocking=True)
    if "hr_mask" in batch:
        out["hr_mask"] = batch["hr_mask"].to(device, non_blocking=True)
    else:
        out["hr_mask"] = None
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    log_interval: int,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    model.train()

    running = {}
    count = 0
    epoch_start = time.time()

    use_amp = scaler is not None and scaler.is_enabled()

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            prediction = model(batch["lr"])
            loss, components = criterion(prediction, batch["hr"], batch["hr_mask"])

        if not math.isfinite(components["total"]):
            print(f"WARNING: non-finite loss at epoch {epoch} step {step}; skipping batch.")
            global_step += 1
            continue

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for key, value in components.items():
            running[key] = running.get(key, 0.0) + value
        count += 1
        global_step += 1

        if log_interval > 0 and (step + 1) % log_interval == 0:
            elapsed = time.time() - epoch_start
            rate = (step + 1) / max(elapsed, 1e-6)
            print(
                f"  epoch {epoch:4d} | step {step + 1:5d}/{len(loader)} "
                f"| loss {components['total']:.5f} "
                f"| {rate:.2f} it/s"
            )

    if count == 0:
        return {"total": float("nan")}, global_step

    averaged = {key: value / count for key, value in running.items()}
    return averaged, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    count = 0
    metric_sums = {key: 0.0 for key in METRIC_KEYS}
    metric_counts = {key: 0 for key in METRIC_KEYS}

    for batch in loader:
        batch = move_batch(batch, device)

        prediction = model(batch["lr"])
        loss, _ = criterion(prediction, batch["hr"], batch["hr_mask"])
        loss_sum += float(loss.item())
        count += 1

        batch_metrics = calculate_all_metrics(
            prediction, batch["hr"], batch["hr_mask"], scale=SCALE
        )
        for key in METRIC_KEYS:
            value = batch_metrics.get(key, float("nan"))
            if value is not None and math.isfinite(value):
                metric_sums[key] += value
                metric_counts[key] += 1

    result = {"val_loss": loss_sum / max(count, 1)}
    for key in METRIC_KEYS:
        result[key] = (
            metric_sums[key] / metric_counts[key]
            if metric_counts[key] > 0
            else float("nan")
        )
    return result


# ============================================================
# HISTORY LOGGING
# ============================================================

def append_history(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SEN2NEON super-resolution model.")

    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml.")

    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--patch-index", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--model", choices=["cnn", "transformer", "hybrid"], default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = merge_defaults(load_config(args.config))
    config = apply_overrides(config, args)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    output_cfg = config["output"]

    set_seed(int(train_cfg["seed"]))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    data_root = Path(data_cfg["data_root"]).resolve()
    patch_index = (
        Path(data_cfg["patch_index"]).resolve()
        if data_cfg.get("patch_index")
        else data_root / "patches" / "all_patches.csv"
    )

    if not patch_index.exists():
        raise FileNotFoundError(
            f"Patch index not found: {patch_index}\n"
            "Run the preprocessing pipeline first: "
            "create_splits.py -> create_patches.py -> normalize.py -> verify_dataset.py"
        )

    output_dir = Path(output_cfg["output_dir"]).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.used.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print("=" * 72)
    print("SEN2NEON TRAINING")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Model           : {model_cfg['name']}")
    print(f"Patch index     : {patch_index}")
    print(f"Output dir      : {output_dir}")
    print()

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    train_loader = create_dataloader(
        patch_index=patch_index,
        data_root=data_root,
        split="train",
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        training=True,
        return_masks=bool(data_cfg["return_masks"]),
        max_samples=data_cfg.get("max_train_samples"),
    )

    val_loader = create_dataloader(
        patch_index=patch_index,
        data_root=data_root,
        split="val",
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        training=False,
        shuffle=False,
        return_masks=bool(data_cfg["return_masks"]),
        max_samples=data_cfg.get("max_val_samples"),
    )

    print(f"Train patches   : {len(train_loader.dataset)}")
    print(f"Val patches     : {len(val_loader.dataset)}")
    print()

    # ------------------------------------------------------------
    # Model / optimizer / scheduler / loss
    # ------------------------------------------------------------
    model_name = model_cfg["name"]
    model, model_config = build_model_and_config(model_name, model_cfg, model_cfg["scale"])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params    : {total_params:,}")
    print()

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    criterion = build_loss(loss_cfg)

    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    higher_is_better = bool(train_cfg["higher_is_better"])
    early_stop_metric = str(train_cfg["early_stopping_metric"])
    patience = int(train_cfg["early_stopping_patience"])

    start_epoch = 0
    global_step = 0
    best_metric = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_epoch, global_step, best_metric = load_checkpoint_for_resume(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        print()

    epochs = int(train_cfg["epochs"])
    val_interval = max(int(train_cfg["val_interval"]), 1)
    epochs_without_improvement = 0

    def is_better(candidate: float, reference: Optional[float]) -> bool:
        if reference is None or not math.isfinite(reference):
            return math.isfinite(candidate)
        if not math.isfinite(candidate):
            return False
        return candidate > reference if higher_is_better else candidate < reference

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            train_stats, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler if use_amp else None,
                grad_clip=float(train_cfg["grad_clip"]),
                log_interval=int(train_cfg["log_interval"]),
                epoch=epoch,
                global_step=global_step,
            )

            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{k}": v for k, v in train_stats.items()})

            val_stats = {}
            if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
                val_stats = validate(model, val_loader, criterion, device)
                row.update(val_stats)

                print(
                    f"[epoch {epoch:4d}] "
                    f"train_loss={train_stats.get('total', float('nan')):.5f} "
                    f"val_loss={val_stats.get('val_loss', float('nan')):.5f} "
                    f"psnr={val_stats.get('psnr_db', float('nan')):.3f} "
                    f"ssim={val_stats.get('ssim', float('nan')):.4f} "
                    f"sam={val_stats.get('sam_degrees', float('nan')):.3f} "
                    f"({time.time() - epoch_start:.1f}s)"
                )

                candidate = val_stats.get(early_stop_metric, float("nan"))
                scheduler_step(scheduler, candidate)

                if is_better(candidate, best_metric):
                    best_metric = candidate
                    epochs_without_improvement = 0
                    if output_cfg["save_best"]:
                        save_checkpoint(
                            checkpoint_dir / "best.pt", model, optimizer, scheduler,
                            epoch, global_step, model_name, model_config, best_metric, config,
                        )
                        print(f"  -> new best ({early_stop_metric}={best_metric:.5f}), saved best.pt")
                else:
                    epochs_without_improvement += 1
            else:
                print(
                    f"[epoch {epoch:4d}] "
                    f"train_loss={train_stats.get('total', float('nan')):.5f} "
                    f"({time.time() - epoch_start:.1f}s)"
                )
                scheduler_step(scheduler, None)

            append_history(output_dir / "history.csv", row)

            if output_cfg["save_last"]:
                save_checkpoint(
                    checkpoint_dir / "last.pt", model, optimizer, scheduler,
                    epoch, global_step, model_name, model_config, best_metric, config,
                )

            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping: no improvement in '{early_stop_metric}' "
                    f"for {patience} validation rounds."
                )
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving last checkpoint before exit...")
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler,
            epoch, global_step, model_name, model_config, best_metric, config,
        )
        print("Saved. Exiting.")
        return

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Best {early_stop_metric}: {best_metric}")
    print(f"Checkpoints in : {checkpoint_dir}")
    print(f"History        : {output_dir / 'history.csv'}")
    print()
    print(
        "Next: evaluate with\n"
        f"  python evaluation/evaluate.py --data-root {data_root} "
        f"--checkpoint {checkpoint_dir / 'best.pt'} --model {model_name}"
    )


if __name__ == "__main__":
    main()
