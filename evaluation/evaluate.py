#!/usr/bin/env python3
"""
SEN2NEON model evaluation.

Evaluates a trained checkpoint on patches listed in:
    SEN2NEON_5GB/patches/test_patches.csv

Supported models:
    - cnn
    - transformer
    - hybrid

Also evaluates the non-learned bicubic reference.

Outputs:
    evaluation/<run_name>/summary.json
    evaluation/<run_name>/patch_metrics.csv
    evaluation/<run_name>/per_band_metrics.csv

Example:
    python evaluation/evaluate.py \
        --data-root SEN2NEON_5GB \
        --checkpoint checkpoints/best.pt \
        --model hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F


# Allow execution from project root or evaluation/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import calculate_all_metrics, per_band_metrics
from models.baseline_cnn import BaselineCNN
from models.proposed_model import HybridCNNTransformerSR
from models.transformer import TransformerSR


LR_NODATA = 65535
HR_NODATA = 0
SCALE = 4
NUM_BANDS = 12
REFLECTANCE_SCALE = 10000.0

BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]


def resolve_path(value: str, data_root: Path) -> Path:
    """Resolve a patch CSV path against the dataset root/project root."""
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        data_root / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the most likely location for a useful error.
    return (data_root / path).resolve()


def first_column(df: pd.DataFrame, names: Iterable[str], required: bool = True) -> Optional[str]:
    """Find the first available column from a list of alternatives."""
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"Could not find any of these columns: {list(names)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def read_patch(
    row: pd.Series,
    data_root: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Read one LR/HR patch directly from GeoTIFF windows."""
    lr_path_col = first_column(row.to_frame().T, ["lr_path", "lr_file", "lr_filepath"])
    hr_path_col = first_column(row.to_frame().T, ["hr_path", "hr_file", "hr_filepath"])

    lr_path = resolve_path(row[lr_path_col], data_root)
    hr_path = resolve_path(row[hr_path_col], data_root)

    lx_col = first_column(row.to_frame().T, ["lr_x", "x_lr", "lr_col", "x"])
    ly_col = first_column(row.to_frame().T, ["lr_y", "y_lr", "lr_row", "y"])
    ls_col = first_column(row.to_frame().T, ["lr_size", "lr_patch_size"], required=False)

    hx_col = first_column(row.to_frame().T, ["hr_x", "x_hr"], required=False)
    hy_col = first_column(row.to_frame().T, ["hr_y", "y_hr"], required=False)
    hs_col = first_column(row.to_frame().T, ["hr_size", "hr_patch_size"], required=False)

    lr_x = int(row[lx_col])
    lr_y = int(row[ly_col])
    lr_size = int(row[ls_col]) if ls_col else 64

    hr_x = int(row[hx_col]) if hx_col else lr_x * SCALE
    hr_y = int(row[hy_col]) if hy_col else lr_y * SCALE
    hr_size = int(row[hs_col]) if hs_col else lr_size * SCALE

    with rasterio.open(lr_path) as src:
        lr_raw = src.read(
            indexes=list(range(1, NUM_BANDS + 1)),
            window=rasterio.windows.Window(
                lr_x, lr_y, lr_size, lr_size
            ),
        )

    with rasterio.open(hr_path) as src:
        hr_raw = src.read(
            indexes=list(range(1, NUM_BANDS + 1)),
            window=rasterio.windows.Window(
                hr_x, hr_y, hr_size, hr_size
            ),
        )

    if lr_raw.shape != (NUM_BANDS, lr_size, lr_size):
        raise RuntimeError(
            f"Unexpected LR patch shape {lr_raw.shape} from {lr_path}"
        )

    if hr_raw.shape != (NUM_BANDS, hr_size, hr_size):
        raise RuntimeError(
            f"Unexpected HR patch shape {hr_raw.shape} from {hr_path}"
        )

    lr_valid = lr_raw != LR_NODATA
    hr_valid = hr_raw != HR_NODATA

    lr = torch.from_numpy(
        lr_raw.astype(np.float32) / REFLECTANCE_SCALE
    )
    hr = torch.from_numpy(
        hr_raw.astype(np.float32) / REFLECTANCE_SCALE
    )

    # Replace invalid source values so they cannot contaminate interpolation.
    lr = torch.where(
        torch.from_numpy(lr_valid),
        lr,
        torch.zeros_like(lr),
    )

    hr = torch.where(
        torch.from_numpy(hr_valid),
        hr,
        torch.zeros_like(hr),
    )

    metadata = {
        "patch_id": str(row.get("patch_id", row.name)),
        "group_id": str(row.get("group_id", "")),
        "lr_path": str(lr_path),
        "hr_path": str(hr_path),
        "lr_x": lr_x,
        "lr_y": lr_y,
        "hr_x": hr_x,
        "hr_y": hr_y,
    }

    return lr, hr, torch.from_numpy(hr_valid), metadata


def build_model(
    model_name: str,
    checkpoint: object,
    device: torch.device,
) -> torch.nn.Module:
    """Construct a model, optionally using checkpoint model_config."""
    config = {}

    if isinstance(checkpoint, dict):
        raw_config = checkpoint.get("model_config", {})
        if isinstance(raw_config, dict):
            config = dict(raw_config)

    # Keep only constructor-compatible options.
    if model_name == "cnn":
        allowed = {
            "in_channels", "out_channels", "features",
            "num_blocks", "scale", "residual_scale",
        }
        config = {k: v for k, v in config.items() if k in allowed}
        model = BaselineCNN(**config)

    elif model_name == "transformer":
        allowed = {
            "in_channels", "out_channels", "embed_dim",
            "depth", "num_heads", "window_size", "scale",
            "dropout",
        }
        config = {k: v for k, v in config.items() if k in allowed}
        model = TransformerSR(**config)

    elif model_name == "hybrid":
        allowed = {
            "in_channels", "out_channels", "features",
            "transformer_dim", "cnn_blocks",
            "transformer_blocks", "heads", "window_size",
            "scale",
        }
        config = {k: v for k, v in config.items() if k in allowed}
        model = HybridCNNTransformerSR(**config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    """Load common checkpoint formats."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
        ]:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint does not contain a recognizable state_dict."
        )

    # Handle DataParallel checkpoints.
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing model parameters. "
            f"First missing keys: {missing[:10]}"
        )

    if unexpected:
        print(
            f"WARNING: ignored {len(unexpected)} unexpected checkpoint keys."
        )

    return checkpoint if isinstance(checkpoint, dict) else {}


def bicubic_prediction(
    lr: torch.Tensor,
) -> torch.Tensor:
    """4x bicubic interpolation of the LR multispectral image."""
    return F.interpolate(
        lr.unsqueeze(0),
        scale_factor=SCALE,
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)


def safe_mean(values: List[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float("nan")


def aggregate_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = rows[0].keys()
    return {
        key: safe_mean([float(row[key]) for row in rows])
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SEN2NEON super-resolution models."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("SEN2NEON_5GB"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained checkpoint. Required unless --bicubic-only is used.",
    )

    parser.add_argument(
        "--model",
        choices=["cnn", "transformer", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
    )

    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--bicubic-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    data_root = args.data_root.resolve()
    patch_csv = data_root / "patches" / "test_patches.csv"

    if not patch_csv.exists():
        raise FileNotFoundError(
            f"Test patch index not found: {patch_csv}"
        )

    df = pd.read_csv(patch_csv)

    if args.max_patches is not None:
        df = df.head(args.max_patches).copy()

    if df.empty:
        raise RuntimeError("Test patch index is empty.")

    model = None
    checkpoint = None

    if not args.bicubic_only:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required unless --bicubic-only is used."
            )

        model = build_model(
            args.model,
            {},
            device,
        )

        checkpoint = load_checkpoint(
            model,
            args.checkpoint.resolve(),
            device,
        )

        model.eval()

    run_name = (
        "bicubic"
        if args.bicubic_only
        else f"{args.model}_{args.checkpoint.stem}"
    )

    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_rows: List[Dict[str, object]] = []
    per_band_rows: List[Dict[str, object]] = []

    print("=" * 80)
    print("SEN2NEON EVALUATION")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Test patches : {len(df)}")
    print(f"Mode         : {'bicubic' if args.bicubic_only else args.model}")
    print()

    with torch.inference_mode():
        for index, row in df.iterrows():
            lr, hr, hr_valid, metadata = read_patch(
                row,
                data_root,
            )

            lr_batch = lr.unsqueeze(0).to(device)
            hr_batch = hr.unsqueeze(0).to(device)
            mask_batch = hr_valid.unsqueeze(0).to(device)

            if args.bicubic_only:
                prediction = bicubic_prediction(
                    lr.to(device)
                ).unsqueeze(0)
            else:
                prediction = model(
                    lr_batch
                )

            if prediction.shape[-2:] != hr_batch.shape[-2:]:
                raise RuntimeError(
                    f"Prediction/HR size mismatch: "
                    f"{tuple(prediction.shape)} vs {tuple(hr_batch.shape)}"
                )

            metrics = calculate_all_metrics(
                prediction,
                hr_batch,
                mask_batch,
                scale=SCALE,
            )

            patch_id = metadata["patch_id"]

            patch_result = {
                "patch_id": patch_id,
                "group_id": metadata["group_id"],
                **metrics,
            }

            patch_rows.append(patch_result)

            band_metrics = per_band_metrics(
                prediction,
                hr_batch,
                mask_batch,
            )

            for band_index, band_name in enumerate(BAND_NAMES):
                per_band_rows.append({
                    "patch_id": patch_id,
                    "band": band_name,
                    "mae": band_metrics["mae"][band_index],
                    "rmse": band_metrics["rmse"][band_index],
                    "psnr_db": band_metrics["psnr"][band_index],
                })

            if (len(patch_rows) % 25 == 0) or len(patch_rows) == len(df):
                print(
                    f"Evaluated {len(patch_rows)}/{len(df)} patches"
                )

    metric_names = [
        "mae",
        "rmse",
        "psnr_db",
        "ssim",
        "sam_degrees",
        "ergas",
        "spectral_correlation",
    ]

    summary_metrics = aggregate_dict([
        {
            key: float(row[key])
            for key in metric_names
        }
        for row in patch_rows
    ])

    band_df = pd.DataFrame(per_band_rows)
    band_summary = {}

    if not band_df.empty:
        for band in BAND_NAMES:
            subset = band_df[band_df["band"] == band]
            band_summary[band] = {
                "mae": float(subset["mae"].mean()),
                "rmse": float(subset["rmse"].mean()),
                "psnr_db": float(subset["psnr_db"].mean()),
            }

    patch_df = pd.DataFrame(patch_rows)

    patch_df.to_csv(
        output_dir / "patch_metrics.csv",
        index=False,
    )

    band_df.to_csv(
        output_dir / "per_band_metrics.csv",
        index=False,
    )

    summary = {
        "model": "bicubic" if args.bicubic_only else args.model,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "data_root": str(data_root),
        "num_test_patches": len(patch_rows),
        "metrics": summary_metrics,
        "per_band": band_summary,
        "scale": SCALE,
        "bands": BAND_NAMES,
        "reflectance_scale": REFLECTANCE_SCALE,
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("RESULTS")
    print("-" * 80)
    for key, value in summary_metrics.items():
        print(f"{key:24s}: {value:.8f}")

    print()
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'patch_metrics.csv'}")
    print(f"Saved: {output_dir / 'per_band_metrics.csv'}")
    print()
    print("STATUS: EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
