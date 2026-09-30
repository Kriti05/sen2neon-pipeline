#!/usr/bin/env python3
"""
SEN2NEON Patch Index Generator
================================

Creates a leakage-safe patch index from the train/val/test split files.

Important:
    This script DOES NOT duplicate image data.

Each patch record stores:
    - source LR GeoTIFF
    - source HR GeoTIFF
    - LR window coordinates
    - HR window coordinates
    - split
    - acquisition/group information
    - nodata percentages
    - optional land-cover metadata

Canonical SEN2NEON:
    LR = 10 m
    HR = 2.5 m
    scale = 4
    LR tile = 256 x 256
    HR tile = 1024 x 1024
    bands = 12

Default:
    LR patch  = 64 x 64
    HR patch  = 256 x 256
    stride    = 64

Therefore:
    256 / 64 = 4 patches per dimension
    4 x 4 = 16 patches per tile

For 378 complete pairs:
    maximum = 378 x 16 = 6048 patches

Later, the PyTorch Dataset will read these windows directly
from the original GeoTIFFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_BANDS = 12

LR_SIZE = 256
HR_SIZE = 1024

DEFAULT_SCALE = 4
DEFAULT_PATCH_SIZE = 64
DEFAULT_STRIDE = 64

LR_NODATA = 65535
HR_NODATA = 0

DEFAULT_MAX_NODATA = 0.05


# ============================================================
# LOGGING
# ============================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an aligned SEN2NEON patch index."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("SEN2NEON_5GB"),
        help="Root directory containing SEN2NEON data.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="LR patch size in pixels. Default: 64.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="LR patch stride in pixels. Default: 64.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="LR -> HR scale factor. Default: 4.",
    )

    parser.add_argument(
        "--max-nodata",
        type=float,
        default=DEFAULT_MAX_NODATA,
        help=(
            "Maximum allowed invalid/nodata fraction per patch. "
            "Default: 0.05."
        ),
    )

    parser.add_argument(
        "--min-valid-bands",
        type=int,
        default=12,
        help=(
            "Minimum number of bands that must contain valid pixels. "
            "Default: 12."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to process.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <data-root>/patches.",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Debug option: process only the first N pairs per split.",
    )

    parser.add_argument(
        "--include-all-nodata",
        action="store_true",
        help="Keep patches even if they contain excessive nodata.",
    )

    return parser.parse_args()


# ============================================================
# PATH HANDLING
# ============================================================

def resolve_path(data_root: Path, value: str) -> Path:
    """
    Resolve a path from split CSV.

    Supports:
        s2_l2a_10m/file.tif
        neon_2.5m_linearized/file.tif
        absolute paths
        relative paths
    """

    path = Path(str(value))

    if path.is_absolute():
        return path

    return data_root / path


# ============================================================
# CSV NORMALIZATION
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    """
    Find the first available column from candidate names.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"None of the required columns were found.\n"
            f"Candidates: {candidates}\n"
            f"Available: {list(df.columns)}"
        )

    return None


def prepare_split_dataframe(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Normalize expected LR/HR path columns.
    """

    lr_col = find_column(
        df,
        [
            "lr_path",
            "lr",
            "lr_file",
            "lr_filepath",
            "s2_path",
        ],
    )

    hr_col = find_column(
        df,
        [
            "hr_path",
            "hr",
            "hr_file",
            "hr_filepath",
            "hr_2_5m_path",
        ],
    )

    result = pd.DataFrame()

    result["lr_path"] = df[lr_col].astype(str)
    result["hr_path"] = df[hr_col].astype(str)

    # --------------------------------------------------------
    # Optional columns
    # --------------------------------------------------------

    optional_columns = [
        "name",
        "id",
        "group_id",
        "acquisition_id",
        "neon_acquisition_id",
        "landcover",
        "land_cover",
        "LC_superclass",
        "LC_superclass_label",
        "LC_detail_label",
        "centroid_lat",
        "centroid_lon",
        "latitude",
        "longitude",
    ]

    for candidate in optional_columns:
        col = find_column(
            df,
            [candidate],
            required=False,
        )

        if col is not None and candidate not in result.columns:
            result[candidate] = df[col]

    result["split"] = split_name

    return result


# ============================================================
# RASTER VALIDATION
# ============================================================

def validate_raster(
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_resolution: float,
    label: str,
) -> Dict:
    """
    Open and validate a SEN2NEON GeoTIFF.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{label} file does not exist: {path}"
        )

    with rasterio.open(path) as src:

        if src.count != EXPECTED_BANDS:
            raise ValueError(
                f"{label} must contain {EXPECTED_BANDS} bands, "
                f"got {src.count}: {path}"
            )

        if src.width != expected_width or src.height != expected_height:
            raise ValueError(
                f"{label} has unexpected dimensions: "
                f"{src.width}x{src.height}; "
                f"expected {expected_width}x{expected_height}: {path}"
            )

        transform = src.transform

        pixel_x = abs(transform.a)
        pixel_y = abs(transform.e)

        if not math.isclose(
            pixel_x,
            expected_resolution,
            rel_tol=0.02,
            abs_tol=0.05,
        ):
            raise ValueError(
                f"{label} X resolution is {pixel_x}, "
                f"expected approximately {expected_resolution}: {path}"
            )

        if not math.isclose(
            pixel_y,
            expected_resolution,
            rel_tol=0.02,
            abs_tol=0.05,
        ):
            raise ValueError(
                f"{label} Y resolution is {pixel_y}, "
                f"expected approximately {expected_resolution}: {path}"
            )

        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "bounds": (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ),
            "nodata": src.nodata,
            "resolution_x": pixel_x,
            "resolution_y": pixel_y,
        }


# ============================================================
# ALIGNMENT CHECK
# ============================================================

def check_alignment(
    lr_meta: Dict,
    hr_meta: Dict,
    scale: int,
) -> None:
    """
    Verify that LR and HR represent the same footprint.

    The HR dimensions should be exactly:
        LR dimensions * scale

    Pixel sizes should satisfy:
        LR resolution / HR resolution == scale
    """

    if hr_meta["width"] != lr_meta["width"] * scale:
        raise ValueError(
            "HR width is not exactly LR width * scale."
        )

    if hr_meta["height"] != lr_meta["height"] * scale:
        raise ValueError(
            "HR height is not exactly LR height * scale."
        )

    lr_rx = lr_meta["resolution_x"]
    hr_rx = hr_meta["resolution_x"]

    lr_ry = lr_meta["resolution_y"]
    hr_ry = hr_meta["resolution_y"]

    if not math.isclose(
        lr_rx / hr_rx,
        scale,
        rel_tol=0.02,
        abs_tol=0.05,
    ):
        raise ValueError(
            f"X resolution ratio is incorrect: "
            f"{lr_rx} / {hr_rx}"
        )

    if not math.isclose(
        lr_ry / hr_ry,
        scale,
        rel_tol=0.02,
        abs_tol=0.05,
    ):
        raise ValueError(
            f"Y resolution ratio is incorrect: "
            f"{lr_ry} / {hr_ry}"
        )

    # Bounds should represent the same geographic footprint.
    for lr_value, hr_value in zip(
        lr_meta["bounds"],
        hr_meta["bounds"],
    ):
        tolerance = max(
            1e-3,
            abs(lr_value) * 1e-8,
        )

        if not math.isclose(
            lr_value,
            hr_value,
            rel_tol=1e-8,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "LR and HR geographic bounds do not match."
            )


# ============================================================
# PATCH GRID
# ============================================================

def generate_windows(
    width: int,
    height: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """
    Generate top-left LR coordinates.

    Coordinates are:
        x = column
        y = row
    """

    if patch_size > width or patch_size > height:
        raise ValueError(
            "Patch size is larger than the source tile."
        )

    positions_x = list(
        range(
            0,
            width - patch_size + 1,
            stride,
        )
    )

    positions_y = list(
        range(
            0,
            height - patch_size + 1,
            stride,
        )
    )

    # --------------------------------------------------------
    # Make sure the last area is not accidentally ignored.
    # --------------------------------------------------------

    last_x = width - patch_size
    last_y = height - patch_size

    if positions_x[-1] != last_x:
        positions_x.append(last_x)

    if positions_y[-1] != last_y:
        positions_y.append(last_y)

    return [
        (x, y)
        for y in positions_y
        for x in positions_x
    ]


# ============================================================
# NODATA ANALYSIS
# ============================================================

def calculate_validity(
    array: np.ndarray,
    nodata_value: int,
) -> Tuple[float, int]:
    """
    Calculate:
        nodata_fraction
        number_of_bands_with_valid_pixels

    array:
        [bands, height, width]
    """

    if array.ndim != 3:
        raise ValueError(
            f"Expected [bands,height,width], got {array.shape}"
        )

    invalid = array == nodata_value

    invalid_fraction = float(
        invalid.mean()
    )

    band_valid = (~invalid).reshape(
        array.shape[0],
        -1,
    ).any(axis=1)

    valid_band_count = int(
        band_valid.sum()
    )

    return invalid_fraction, valid_band_count


# ============================================================
# PATCH INSPECTION
# ============================================================

def inspect_patch(
    lr_src,
    hr_src,
    x: int,
    y: int,
    patch_size: int,
    scale: int,
    max_nodata: float,
    min_valid_bands: int,
) -> Dict:
    """
    Read one aligned LR/HR patch and inspect validity.

    This does NOT retain the pixel data.
    """

    lr_window = Window(
        col_off=x,
        row_off=y,
        width=patch_size,
        height=patch_size,
    )

    hr_x = x * scale
    hr_y = y * scale

    hr_size = patch_size * scale

    hr_window = Window(
        col_off=hr_x,
        row_off=hr_y,
        width=hr_size,
        height=hr_size,
    )

    lr = lr_src.read(
        window=lr_window,
    )

    hr = hr_src.read(
        window=hr_window,
    )

    expected_lr_shape = (
        EXPECTED_BANDS,
        patch_size,
        patch_size,
    )

    expected_hr_shape = (
        EXPECTED_BANDS,
        hr_size,
        hr_size,
    )

    if lr.shape != expected_lr_shape:
        raise ValueError(
            f"Unexpected LR patch shape: {lr.shape}; "
            f"expected {expected_lr_shape}"
        )

    if hr.shape != expected_hr_shape:
        raise ValueError(
            f"Unexpected HR patch shape: {hr.shape}; "
            f"expected {expected_hr_shape}"
        )

    lr_nodata_fraction, lr_valid_bands = calculate_validity(
        lr,
        LR_NODATA,
    )

    hr_nodata_fraction, hr_valid_bands = calculate_validity(
        hr,
        HR_NODATA,
    )

    keep = (
        lr_nodata_fraction <= max_nodata
        and hr_nodata_fraction <= max_nodata
        and lr_valid_bands >= min_valid_bands
        and hr_valid_bands >= min_valid_bands
    )

    return {
        "lr_x": x,
        "lr_y": y,
        "lr_size": patch_size,

        "hr_x": hr_x,
        "hr_y": hr_y,
        "hr_size": hr_size,

        "lr_nodata_fraction": lr_nodata_fraction,
        "hr_nodata_fraction": hr_nodata_fraction,

        "lr_valid_bands": lr_valid_bands,
        "hr_valid_bands": hr_valid_bands,

        "keep": bool(keep),
    }


# ============================================================
# PROCESS ONE TILE
# ============================================================

def process_pair(
    row: pd.Series,
    data_root: Path,
    split: str,
    patch_size: int,
    stride: int,
    scale: int,
    max_nodata: float,
    min_valid_bands: int,
    include_all_nodata: bool,
    pair_number: int,
) -> Tuple[List[Dict], Dict]:
    """
    Generate patch records for one LR/HR pair.
    """

    lr_path = resolve_path(
        data_root,
        row["lr_path"],
    )

    hr_path = resolve_path(
        data_root,
        row["hr_path"],
    )

    lr_meta = validate_raster(
        lr_path,
        expected_width=LR_SIZE,
        expected_height=LR_SIZE,
        expected_resolution=10.0,
        label="LR",
    )

    hr_meta = validate_raster(
        hr_path,
        expected_width=HR_SIZE,
        expected_height=HR_SIZE,
        expected_resolution=2.5,
        label="HR",
    )

    check_alignment(
        lr_meta,
        hr_meta,
        scale,
    )

    windows = generate_windows(
        width=LR_SIZE,
        height=LR_SIZE,
        patch_size=patch_size,
        stride=stride,
    )

    records: List[Dict] = []

    accepted = 0
    rejected = 0

    with rasterio.open(lr_path) as lr_src, \
            rasterio.open(hr_path) as hr_src:

        for patch_number, (x, y) in enumerate(
            windows,
            start=1,
        ):

            result = inspect_patch(
                lr_src=lr_src,
                hr_src=hr_src,
                x=x,
                y=y,
                patch_size=patch_size,
                scale=scale,
                max_nodata=max_nodata,
                min_valid_bands=min_valid_bands,
            )

            keep = (
                result["keep"]
                or include_all_nodata
            )

            if not keep:
                rejected += 1
                continue

            accepted += 1

            source_name = Path(
                str(row["lr_path"])
            ).stem

            patch_id = (
                f"{split}__"
                f"{source_name}__"
                f"x{x:04d}_y{y:04d}"
            )

            record = {
                "patch_id": patch_id,
                "split": split,

                "pair_number": pair_number,
                "patch_number": patch_number,

                "lr_path": str(
                    row["lr_path"]
                ),
                "hr_path": str(
                    row["hr_path"]
                ),

                "lr_x": result["lr_x"],
                "lr_y": result["lr_y"],
                "lr_size": result["lr_size"],

                "hr_x": result["hr_x"],
                "hr_y": result["hr_y"],
                "hr_size": result["hr_size"],

                "scale": scale,

                "lr_nodata_fraction": (
                    result["lr_nodata_fraction"]
                ),
                "hr_nodata_fraction": (
                    result["hr_nodata_fraction"]
                ),

                "lr_valid_bands": (
                    result["lr_valid_bands"]
                ),
                "hr_valid_bands": (
                    result["hr_valid_bands"]
                ),
            }

            # ------------------------------------------------
            # Carry optional metadata from split CSV.
            # ------------------------------------------------

            for column in row.index:

                if column in {
                    "lr_path",
                    "hr_path",
                    "split",
                }:
                    continue

                value = row[column]

                if pd.isna(value):
                    value = ""

                record[column] = value

            records.append(record)

    summary = {
        "split": split,
        "source_pair": str(row["lr_path"]),
        "total_grid_patches": len(windows),
        "accepted_patches": accepted,
        "rejected_patches": rejected,
    }

    return records, summary


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split: str,
    data_root: Path,
    output_dir: Path,
    patch_size: int,
    stride: int,
    scale: int,
    max_nodata: float,
    min_valid_bands: int,
    include_all_nodata: bool,
    max_pairs: Optional[int],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process one train/val/test split.
    """

    split_path = (
        data_root
        / "splits"
        / f"{split}.csv"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}"
        )

    log()
    log("=" * 72)
    log(f"PROCESSING SPLIT: {split.upper()}")
    log("=" * 72)
    log(f"Split file: {split_path}")

    df = pd.read_csv(split_path)

    df = prepare_split_dataframe(
        df,
        split_name=split,
    )

    if max_pairs is not None:
        df = df.head(max_pairs).copy()

    log(f"Pairs: {len(df)}")

    all_records: List[Dict] = []
    pair_summaries: List[Dict] = []

    failures = 0

    for pair_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        try:
            records, summary = process_pair(
                row=row,
                data_root=data_root,
                split=split,
                patch_size=patch_size,
                stride=stride,
                scale=scale,
                max_nodata=max_nodata,
                min_valid_bands=min_valid_bands,
                include_all_nodata=include_all_nodata,
                pair_number=pair_number,
            )

            all_records.extend(records)
            pair_summaries.append(summary)

            log(
                f"[{pair_number:4d}/{len(df):4d}] "
                f"{Path(str(row['lr_path'])).name} "
                f"-> {summary['accepted_patches']} patches"
            )

        except Exception as exc:

            failures += 1

            log(
                f"[FAILED {pair_number:4d}/{len(df):4d}] "
                f"{row['lr_path']}"
            )

            log(
                f"    {type(exc).__name__}: {exc}"
            )

    patch_df = pd.DataFrame(
        all_records
    )

    output_path = (
        output_dir
        / f"{split}_patches.csv"
    )

    patch_df.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "split": split,
        "pairs_requested": int(len(df)),
        "pairs_failed": int(failures),
        "pairs_successful": int(
            len(df) - failures
        ),
        "patches_created": int(
            len(patch_df)
        ),
        "patch_size_lr": patch_size,
        "patch_size_hr": patch_size * scale,
        "stride_lr": stride,
        "scale": scale,
        "max_nodata": max_nodata,
        "min_valid_bands": min_valid_bands,
        "pair_summaries": pair_summaries,
    }

    summary_path = (
        output_dir
        / f"{split}_patches_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    log()
    log(
        f"{split.upper()} patches: "
        f"{len(patch_df)}"
    )
    log(
        f"Saved: {output_path}"
    )

    return patch_df, summary


# ============================================================
# COMBINE PATCHES
# ============================================================

def create_combined_index(
    patch_tables: List[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Combine train/val/test patch indexes.
    """

    non_empty = [
        df
        for df in patch_tables
        if not df.empty
    ]

    if not non_empty:
        raise RuntimeError(
            "No patches were created."
        )

    combined = pd.concat(
        non_empty,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Critical leakage check.
    # --------------------------------------------------------

    if "group_id" in combined.columns:

        group_split_counts = (
            combined
            .groupby("group_id")["split"]
            .nunique()
        )

        leaked_groups = (
            group_split_counts[
                group_split_counts > 1
            ]
        )

        if len(leaked_groups) > 0:
            raise RuntimeError(
                "DATA LEAKAGE DETECTED: "
                f"{len(leaked_groups)} groups "
                "appear in multiple splits."
            )

    output_path = (
        output_dir
        / "all_patches.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return combined


# ============================================================
# SUMMARY
# ============================================================

def print_final_summary(
    combined: pd.DataFrame,
    summaries: List[Dict],
    output_dir: Path,
) -> None:

    log()
    log()
    log("=" * 72)
    log("PATCH GENERATION COMPLETE")
    log("=" * 72)

    log(
        f"Total patches: "
        f"{len(combined)}"
    )

    if "split" in combined.columns:

        counts = (
            combined["split"]
            .value_counts()
            .to_dict()
        )

        log(
            f"Train patches: "
            f"{counts.get('train', 0)}"
        )

        log(
            f"Val patches:   "
            f"{counts.get('val', 0)}"
        )

        log(
            f"Test patches:  "
            f"{counts.get('test', 0)}"
        )

    log()
    log(
        f"LR patch: "
        f"{combined['lr_size'].iloc[0]}x"
        f"{combined['lr_size'].iloc[0]}"
    )

    log(
        f"HR patch: "
        f"{combined['hr_size'].iloc[0]}x"
        f"{combined['hr_size'].iloc[0]}"
    )

    log(
        f"Scale: "
        f"{combined['scale'].iloc[0]}x"
    )

    log()
    log(
        f"Output directory: "
        f"{output_dir}"
    )

    log(
        f"Combined index: "
        f"{output_dir / 'all_patches.csv'}"
    )

    log()
    log("STATUS: PATCH INDEX READY")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.patch_size <= 0:
        fail("patch-size must be > 0.")

    if args.stride <= 0:
        fail("stride must be > 0.")

    if args.scale <= 0:
        fail("scale must be > 0.")

    if not 0.0 <= args.max_nodata <= 1.0:
        fail(
            "max-nodata must be between 0 and 1."
        )

    if not (
        1 <= args.min_valid_bands <= EXPECTED_BANDS
    ):
        fail(
            f"min-valid-bands must be between "
            f"1 and {EXPECTED_BANDS}."
        )

    if LR_SIZE % args.patch_size != 0:
        log(
            "WARNING: patch size does not evenly divide "
            "the 256x256 LR tile. "
            "The script will still cover the final edge."
        )

    expected_hr_size = (
        args.patch_size * args.scale
    )

    log("=" * 72)
    log("SEN2NEON PATCH INDEX GENERATOR")
    log("=" * 72)

    log(
        f"Data root       : {args.data_root}"
    )

    log(
        f"LR patch        : "
        f"{args.patch_size} x {args.patch_size}"
    )

    log(
        f"HR patch        : "
        f"{expected_hr_size} x {expected_hr_size}"
    )

    log(
        f"Scale           : {args.scale}x"
    )

    log(
        f"Stride          : {args.stride}"
    )

    log(
        f"Max nodata      : "
        f"{args.max_nodata:.2%}"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.data_root / "patches"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_tables: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    for split in args.splits:

        patch_df, summary = process_split(
            split=split,
            data_root=args.data_root,
            output_dir=output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            scale=args.scale,
            max_nodata=args.max_nodata,
            min_valid_bands=args.min_valid_bands,
            include_all_nodata=args.include_all_nodata,
            max_pairs=args.max_pairs,
        )

        patch_tables.append(patch_df)
        summaries.append(summary)

    # --------------------------------------------------------
    # Combined index
    # --------------------------------------------------------

    combined = create_combined_index(
        patch_tables,
        output_dir,
    )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    global_summary = {
        "dataset": "SEN2NEON",
        "bands": EXPECTED_BANDS,

        "lr_resolution_m": 10.0,
        "hr_resolution_m": 2.5,

        "scale": args.scale,

        "source_lr_size": [
            LR_SIZE,
            LR_SIZE,
        ],

        "source_hr_size": [
            HR_SIZE,
            HR_SIZE,
        ],

        "patch_lr_size": [
            args.patch_size,
            args.patch_size,
        ],

        "patch_hr_size": [
            expected_hr_size,
            expected_hr_size,
        ],

        "stride_lr": args.stride,

        "max_nodata_fraction": (
            args.max_nodata
        ),

        "min_valid_bands": (
            args.min_valid_bands
        ),

        "total_patches": int(
            len(combined)
        ),

        "splits": summaries,
    }

    summary_path = (
        output_dir
        / "patch_generation_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            global_summary,
            f,
            indent=2,
        )

    print_final_summary(
        combined=combined,
        summaries=summaries,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()