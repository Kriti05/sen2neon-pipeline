#!/usr/bin/env python3
"""
SEN2NEON qualitative visualization.

Creates publication/debug-friendly comparison images for one test patch:

    LR bicubic RGB
    prediction RGB
    HR RGB
    absolute error RGB

and a false-color NIR comparison:

    B8 / B4 / B3

It also saves individual per-band error maps and a JSON metrics file.

Example:
    python evaluation/visualize.py \
        --data-root SEN2NEON_5GB \
        --checkpoint checkpoints/best.pt \
        --model hybrid \
        --patch-index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (
    BAND_NAMES,
    build_model,
    load_checkpoint,
    read_patch,
)
from evaluation.metrics import calculate_all_metrics


SCALE = 4

# Sentinel-2 band positions in the 12-band SEN2NEON ordering.
# B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
RGB = (3, 2, 1)       # B4 / B3 / B2
FALSE_COLOR = (7, 3, 2)  # B8 / B4 / B3


def robust_normalize(
    image: np.ndarray,
    valid: Optional[np.ndarray] = None,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """Per-channel percentile stretch for visualization only."""
    image = image.astype(np.float32, copy=False)

    output = np.zeros_like(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {image.shape}"
        )

    for channel in range(image.shape[0]):
        band = image[channel]

        if valid is None:
            values = band[np.isfinite(band)]
        else:
            values = band[valid[channel] & np.isfinite(band)]

        if values.size == 0:
            continue

        lo, hi = np.percentile(
            values,
            [low, high],
        )

        if hi <= lo:
            output[channel] = 0.0
        else:
            output[channel] = np.clip(
                (band - lo) / (hi - lo),
                0.0,
                1.0,
            )

    return output


def make_rgb(
    image: np.ndarray,
    band_indices: Tuple[int, int, int],
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create HxWx3 visualization image from CxHxW array."""
    selected = image[list(band_indices)]

    selected_valid = None
    if valid is not None:
        selected_valid = valid[list(band_indices)]

    normalized = robust_normalize(
        selected,
        selected_valid,
    )

    return np.transpose(
        normalized,
        (1, 2, 0),
    )


def error_rgb(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Normalize absolute RGB error for display."""
    error = np.abs(
        prediction[list(RGB)]
        - target[list(RGB)]
    )

    # Error is visualized independently from reflectance stretch.
    return make_rgb(
        error,
        (0, 1, 2),
    )


def save_comparison_figure(
    lr: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str,
    lr_valid: Optional[np.ndarray] = None,
    hr_valid: Optional[np.ndarray] = None,
) -> None:
    """Save RGB + false-color comparison figure."""
    lr_rgb = make_rgb(
        lr,
        RGB,
        lr_valid,
    )

    pred_rgb = make_rgb(
        prediction,
        RGB,
    )

    target_rgb = make_rgb(
        target,
        RGB,
        hr_valid,
    )

    lr_fc = make_rgb(
        lr,
        FALSE_COLOR,
        lr_valid,
    )

    pred_fc = make_rgb(
        prediction,
        FALSE_COLOR,
    )

    target_fc = make_rgb(
        target,
        FALSE_COLOR,
        hr_valid,
    )

    rgb_error = error_rgb(
        prediction,
        target,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    panels = [
        (axes[0, 0], lr_rgb, "LR → bicubic RGB"),
        (axes[0, 1], pred_rgb, "Prediction RGB"),
        (axes[0, 2], target_rgb, "HR target RGB"),
        (axes[0, 3], rgb_error, "Absolute RGB error"),
        (axes[1, 0], lr_fc, "LR → bicubic NIR"),
        (axes[1, 1], pred_fc, "Prediction NIR"),
        (axes[1, 2], target_fc, "HR target NIR"),
    ]

    for axis, image, panel_title in panels:
        axis.imshow(image)
        axis.set_title(panel_title)
        axis.axis("off")

    # Mean absolute error image across all 12 bands.
    mean_error = np.mean(
        np.abs(prediction - target),
        axis=0,
    )

    axes[1, 3].imshow(
        mean_error,
        cmap="magma",
    )
    axes[1, 3].set_title("Mean spectral absolute error")
    axes[1, 3].axis("off")

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_band_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    output_dir: Path,
) -> None:
    """Save individual band error maps."""
    error = np.abs(
        prediction - target
    )

    for index, band_name in enumerate(BAND_NAMES):
        band_error = error[index].copy()
        band_error[~valid[index]] = np.nan

        fig, axis = plt.subplots(
            figsize=(6, 5),
        )

        image = axis.imshow(
            band_error,
            cmap="magma",
        )

        axis.set_title(
            f"Absolute error — {band_name}"
        )
        axis.axis("off")

        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"error_{band_name}.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a SEN2NEON super-resolution prediction."
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
    )

    parser.add_argument(
        "--model",
        choices=["cnn", "transformer", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--patch-index",
        type=int,
        default=0,
        help="Row index in test_patches.csv.",
    )

    parser.add_argument(
        "--patch-id",
        type=str,
        default=None,
        help="Exact patch_id. Overrides --patch-index.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--bicubic-only",
        action="store_true",
        help="Visualize bicubic instead of a model checkpoint.",
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

    if df.empty:
        raise RuntimeError("Test patch index is empty.")

    if args.patch_id is not None:
        if "patch_id" not in df.columns:
            raise KeyError("test_patches.csv does not contain patch_id")

        matches = df[df["patch_id"].astype(str) == args.patch_id]
        if matches.empty:
            raise ValueError(
                f"Patch ID not found: {args.patch_id}"
            )
        row = matches.iloc[0]
    else:
        if args.patch_index < 0 or args.patch_index >= len(df):
            raise IndexError(
                f"patch-index must be between 0 and {len(df)-1}"
            )
        row = df.iloc[args.patch_index]

    lr, hr, hr_valid, metadata = read_patch(
        row,
        data_root,
    )

    # LR validity is reconstructed from the source values in this script.
    # Invalid LR pixels are not used in the visualization stretch.
    lr_valid = torch.isfinite(lr) & (lr >= 0)

    if args.bicubic_only:
        with torch.inference_mode():
            prediction = F.interpolate(
                lr.unsqueeze(0).to(device),
                scale_factor=SCALE,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).cpu()
        run_name = "bicubic"
    else:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required unless --bicubic-only is used."
            )

        # Read model config from checkpoint before construction.
        checkpoint_raw = torch.load(
            args.checkpoint.resolve(),
            map_location=device,
            weights_only=False,
        )

        model = build_model(
            args.model,
            checkpoint_raw,
            device,
        )

        load_checkpoint(
            model,
            args.checkpoint.resolve(),
            device,
        )

        model.eval()

        with torch.inference_mode():
            prediction = model(
                lr.unsqueeze(0).to(device)
            ).squeeze(0).cpu()

        run_name = f"{args.model}_{args.checkpoint.stem}"

    prediction_np = prediction.numpy()
    target_np = hr.numpy()
    lr_np = lr.numpy()
    valid_np = hr_valid.numpy().astype(bool)
    lr_valid_np = lr_valid.numpy().astype(bool)

    output_dir = (
        args.output_dir
        / run_name
        / str(metadata["patch_id"])
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = calculate_all_metrics(
        prediction.unsqueeze(0),
        hr.unsqueeze(0),
        hr_valid.unsqueeze(0),
        scale=SCALE,
    )

    save_comparison_figure(
        lr=F.interpolate(
            lr.unsqueeze(0),
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).numpy(),
        prediction=prediction_np,
        target=target_np,
        output_path=output_dir / "comparison.png",
        title=(
            f"SEN2NEON | {metadata['patch_id']} | {run_name}"
        ),
        lr_valid=F.interpolate(
            lr_valid.float().unsqueeze(0),
            scale_factor=SCALE,
            mode="nearest",
        ).squeeze(0).numpy().astype(bool),
        hr_valid=valid_np,
    )

    save_band_errors(
        prediction_np,
        target_np,
        valid_np,
        output_dir,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "patch_id": metadata["patch_id"],
                "model": run_name,
                "metrics": metrics,
                "bands": BAND_NAMES,
            },
            handle,
            indent=2,
        )

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Patch : {metadata['patch_id']}")
    print(f"Model : {run_name}")
    print()

    for key, value in metrics.items():
        print(f"{key:24s}: {value:.8f}")

    print()
    print(f"Comparison : {output_dir / 'comparison.png'}")
    print(f"Errors     : {output_dir}")
    print(f"Metrics    : {output_dir / 'metrics.json'}")
    print()
    print("STATUS: VISUALIZATION READY")


if __name__ == "__main__":
    main()
