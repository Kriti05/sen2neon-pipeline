#!/usr/bin/env python3
"""
SEN2NEON Dataset Verification
=============================

Verifies a local SEN2NEON 10m -> 2.5m paired dataset.

Expected canonical structure:

DATA_ROOT/
├── s2_l2a_10m/
│   └── *.tif
└── neon_2.5m_linearized/
    └── *.tif

Expected:
    LR: 12 x 256 x 256 @ 10m
    HR: 12 x 1024 x 1024 @ 2.5m

The script performs:
    - pair matching
    - TIFF readability checks
    - dimensions
    - band count
    - dtype
    - CRS
    - pixel resolution
    - transform/alignment
    - nodata validation
    - NaN/Inf checks
    - reflectance range checks
    - nodata percentage estimation
    - CSV report
    - JSON summary

Usage:

    python preprocessing/verify_dataset.py --data-root SEN2NEON_5GB

Or:

    python preprocessing/verify_dataset.py \
        --data-root D:/1UPDATE/SIH2/SEN2NEON_5GB
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
except ImportError:
    print(
        "ERROR: rasterio is not installed.\n"
        "Install it with:\n"
        "    pip install rasterio"
    )
    sys.exit(1)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_BANDS = 12

LR_WIDTH = 256
LR_HEIGHT = 256
LR_RESOLUTION = 10.0

HR_WIDTH = 1024
HR_HEIGHT = 1024
HR_RESOLUTION = 2.5

EXPECTED_SCALE = 4.0

EXPECTED_LR_DTYPE = "uint16"
EXPECTED_HR_DTYPE = "uint16"

EXPECTED_LR_NODATA = 65535
EXPECTED_HR_NODATA = 0

REFLECTANCE_SCALE = 10000.0

# Small tolerance for floating-point geotransform values.
TRANSFORM_TOLERANCE = 1e-5

# Number of pixels processed per block when checking values.
BLOCK_SIZE = 512

TIFF_EXTENSIONS = {".tif", ".tiff"}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a local SEN2NEON 10m -> 2.5m dataset."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="SEN2NEON_5GB",
        help="Root directory containing s2_l2a_10m and "
             "neon_2.5m_linearized.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for verification reports. "
             "Default: <data-root>/verification",
    )

    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help=(
            "Pixel sampling step for value checks. "
            "1 = every pixel, 2 = every second pixel, etc. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for testing. "
             "Example: --max-files 10",
    )

    parser.add_argument(
        "--skip-value-check",
        action="store_true",
        help=(
            "Skip pixel-value/NaN/Inf/nodata percentage checks. "
            "Structural checks are still performed."
        ),
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def nearly_equal(a: float, b: float, tol: float = TRANSFORM_TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= tol


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
        return None
    except (TypeError, ValueError):
        return None


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def list_tiffs(directory: Path) -> dict[str, Path]:
    """
    Returns:
        {
            filename.tif: full_path
        }
    """
    if not directory.exists():
        return {}

    files = {}

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS:
            files[path.name] = path

    return files


def compare_transform(lr: rasterio.Affine,
                      hr: rasterio.Affine,
                      lr_res: float,
                      hr_res: float) -> tuple[bool, list[str]]:
    """
    Verify that LR and HR grids have the same spatial origin
    and the expected 4x resolution relationship.
    """

    errors = []

    # Pixel sizes.
    if not nearly_equal(abs(lr.a), LR_RESOLUTION):
        errors.append(
            f"LR pixel width={lr.a}, expected ±{LR_RESOLUTION}"
        )

    if not nearly_equal(abs(lr.e), LR_RESOLUTION):
        errors.append(
            f"LR pixel height={lr.e}, expected ±{LR_RESOLUTION}"
        )

    if not nearly_equal(abs(hr.a), HR_RESOLUTION):
        errors.append(
            f"HR pixel width={hr.a}, expected ±{HR_RESOLUTION}"
        )

    if not nearly_equal(abs(hr.e), HR_RESOLUTION):
        errors.append(
            f"HR pixel height={hr.e}, expected ±{HR_RESOLUTION}"
        )

    # Rotation/shear should be zero for this benchmark.
    for name, value in (
        ("LR transform b", lr.b),
        ("LR transform d", lr.d),
        ("HR transform b", hr.b),
        ("HR transform d", hr.d),
    ):
        if not nearly_equal(value, 0.0):
            errors.append(f"{name}={value}, expected 0")

    # Same upper-left origin.
    if not nearly_equal(lr.c, hr.c):
        errors.append(
            f"origin X mismatch: LR={lr.c}, HR={hr.c}"
        )

    if not nearly_equal(lr.f, hr.f):
        errors.append(
            f"origin Y mismatch: LR={lr.f}, HR={hr.f}"
        )

    # Expected scale.
    if lr_res > 0 and hr_res > 0:
        scale_x = lr_res / hr_res
        scale_y = lr_res / hr_res

        if not nearly_equal(scale_x, EXPECTED_SCALE):
            errors.append(
                f"X scale={scale_x}, expected {EXPECTED_SCALE}"
            )

        if not nearly_equal(scale_y, EXPECTED_SCALE):
            errors.append(
                f"Y scale={scale_y}, expected {EXPECTED_SCALE}"
            )

    return len(errors) == 0, errors


def update_value_stats(
    array: np.ndarray,
    nodata: float | int | None,
    stats: dict[str, Any],
    sample_step: int,
) -> None:
    """
    Update statistics for one raster block.

    The array is expected to be:
        bands x height x width
    """

    if sample_step > 1:
        array = array[
            :,
            ::sample_step,
            ::sample_step,
        ]

    if array.size == 0:
        return

    # Convert to float64 only for analysis.
    values = array.astype(np.float64, copy=False)

    stats["pixels_checked"] += int(values.size)

    finite_mask = np.isfinite(values)

    stats["nan_pixels"] += int(np.isnan(values).sum())
    stats["inf_pixels"] += int(np.isinf(values).sum())

    if not finite_mask.any():
        return

    finite = values[finite_mask]

    stats["min_value"] = min(
        stats["min_value"],
        float(np.min(finite)),
    )

    stats["max_value"] = max(
        stats["max_value"],
        float(np.max(finite)),
    )

    # uint16 reflectance values should normally be within:
    # 0 ... 10000
    #
    # LR nodata=65535 is handled separately.
    if nodata is not None:
        nodata_mask = values == float(nodata)
        stats["nodata_pixels"] += int(nodata_mask.sum())

        valid_mask = finite_mask & ~nodata_mask
    else:
        valid_mask = finite_mask

    valid = values[valid_mask]

    if valid.size:
        stats["valid_pixels"] += int(valid.size)

        stats["below_zero_pixels"] += int(
            np.sum(valid < 0)
        )

        stats["above_reflectance_pixels"] += int(
            np.sum(valid > REFLECTANCE_SCALE)
        )


def inspect_raster(
    path: Path,
    expected_kind: str,
    skip_value_check: bool,
    sample_step: int,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "errors": [],
        "warnings": [],
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "crs": None,
        "pixel_width": None,
        "pixel_height": None,
        "nodata": None,
        "bounds": None,
        "transform": None,
        "nodata_percent": None,
        "nan_pixels": 0,
        "inf_pixels": 0,
        "below_zero_pixels": 0,
        "above_reflectance_pixels": 0,
        "min_value": None,
        "max_value": None,
        "pixels_checked": 0,
    }

    try:
        with rasterio.open(path) as src:

            result["readable"] = True

            result["width"] = src.width
            result["height"] = src.height
            result["bands"] = src.count
            result["dtype"] = str(src.dtypes[0])
            result["crs"] = str(src.crs) if src.crs else None

            result["pixel_width"] = abs(float(src.transform.a))
            result["pixel_height"] = abs(float(src.transform.e))

            result["nodata"] = (
                safe_float(src.nodata)
                if src.nodata is not None
                else None
            )

            result["bounds"] = [
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            ]

            result["transform"] = [
                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),
            ]

            # ------------------------------------------------
            # Structural checks
            # ------------------------------------------------

            if src.count != EXPECTED_BANDS:
                result["errors"].append(
                    f"band count={src.count}, "
                    f"expected {EXPECTED_BANDS}"
                )

            if expected_kind == "LR":

                if src.width != LR_WIDTH:
                    result["errors"].append(
                        f"width={src.width}, expected {LR_WIDTH}"
                    )

                if src.height != LR_HEIGHT:
                    result["errors"].append(
                        f"height={src.height}, expected {LR_HEIGHT}"
                    )

                if result["dtype"] != EXPECTED_LR_DTYPE:
                    result["errors"].append(
                        f"dtype={result['dtype']}, "
                        f"expected {EXPECTED_LR_DTYPE}"
                    )

                if src.nodata is not None:
                    if not nearly_equal(
                        float(src.nodata),
                        EXPECTED_LR_NODATA,
                    ):
                        result["warnings"].append(
                            f"nodata={src.nodata}, "
                            f"expected {EXPECTED_LR_NODATA}"
                        )

            elif expected_kind == "HR":

                if src.width != HR_WIDTH:
                    result["errors"].append(
                        f"width={src.width}, expected {HR_WIDTH}"
                    )

                if src.height != HR_HEIGHT:
                    result["errors"].append(
                        f"height={src.height}, expected {HR_HEIGHT}"
                    )

                if result["dtype"] != EXPECTED_HR_DTYPE:
                    result["errors"].append(
                        f"dtype={result['dtype']}, "
                        f"expected {EXPECTED_HR_DTYPE}"
                    )

                if src.nodata is not None:
                    if not nearly_equal(
                        float(src.nodata),
                        EXPECTED_HR_NODATA,
                    ):
                        result["warnings"].append(
                            f"nodata={src.nodata}, "
                            f"expected {EXPECTED_HR_NODATA}"
                        )

            # ------------------------------------------------
            # Pixel value inspection
            # ------------------------------------------------

            stats = {
                "pixels_checked": 0,
                "valid_pixels": 0,
                "nodata_pixels": 0,
                "nan_pixels": 0,
                "inf_pixels": 0,
                "below_zero_pixels": 0,
                "above_reflectance_pixels": 0,
                "min_value": float("inf"),
                "max_value": float("-inf"),
            }

            if not skip_value_check:

                for _, window in src.block_windows(1):

                    block = src.read(
                        window=window,
                        out_dtype="float32",
                    )

                    update_value_stats(
                        block,
                        src.nodata,
                        stats,
                        sample_step,
                    )

            if stats["pixels_checked"] > 0:

                result["pixels_checked"] = stats["pixels_checked"]

                result["nan_pixels"] = stats["nan_pixels"]
                result["inf_pixels"] = stats["inf_pixels"]

                result["below_zero_pixels"] = (
                    stats["below_zero_pixels"]
                )

                result["above_reflectance_pixels"] = (
                    stats["above_reflectance_pixels"]
                )

                result["min_value"] = (
                    None
                    if stats["min_value"] == float("inf")
                    else stats["min_value"]
                )

                result["max_value"] = (
                    None
                    if stats["max_value"] == float("-inf")
                    else stats["max_value"]
                )

                result["nodata_percent"] = (
                    100.0
                    * stats["nodata_pixels"]
                    / stats["pixels_checked"]
                )

                if result["nan_pixels"] > 0:
                    result["errors"].append(
                        f"NaN pixels={result['nan_pixels']}"
                    )

                if result["inf_pixels"] > 0:
                    result["errors"].append(
                        f"Inf pixels={result['inf_pixels']}"
                    )

                if result["below_zero_pixels"] > 0:
                    result["warnings"].append(
                        f"negative pixels="
                        f"{result['below_zero_pixels']}"
                    )

                if result["above_reflectance_pixels"] > 0:
                    result["warnings"].append(
                        f"values > {int(REFLECTANCE_SCALE)}="
                        f"{result['above_reflectance_pixels']}"
                    )

    except Exception as exc:
        result["errors"].append(
            f"FAILED TO READ TIFF: {type(exc).__name__}: {exc}"
        )

    return result


def check_pair(
    filename: str,
    lr_path: Path,
    hr_path: Path,
    data_root: Path,
    skip_value_check: bool,
    sample_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:

    lr = inspect_raster(
        lr_path,
        "LR",
        skip_value_check,
        sample_step,
    )

    hr = inspect_raster(
        hr_path,
        "HR",
        skip_value_check,
        sample_step,
    )

    pair_errors: list[str] = []
    pair_warnings: list[str] = []

    # Only perform spatial checks when both files opened.
    if lr["readable"] and hr["readable"]:

        # CRS.
        if lr["crs"] != hr["crs"]:
            pair_errors.append(
                f"CRS mismatch: LR={lr['crs']} "
                f"HR={hr['crs']}"
            )

        # Resolution.
        if not nearly_equal(
            lr["pixel_width"],
            LR_RESOLUTION,
        ):
            pair_errors.append(
                f"LR X resolution={lr['pixel_width']}"
            )

        if not nearly_equal(
            lr["pixel_height"],
            LR_RESOLUTION,
        ):
            pair_errors.append(
                f"LR Y resolution={lr['pixel_height']}"
            )

        if not nearly_equal(
            hr["pixel_width"],
            HR_RESOLUTION,
        ):
            pair_errors.append(
                f"HR X resolution={hr['pixel_width']}"
            )

        if not nearly_equal(
            hr["pixel_height"],
            HR_RESOLUTION,
        ):
            pair_errors.append(
                f"HR Y resolution={hr['pixel_height']}"
            )

        # Transform/alignment.
        with rasterio.open(lr_path) as lr_src, \
             rasterio.open(hr_path) as hr_src:

            transform_ok, transform_errors = compare_transform(
                lr_src.transform,
                hr_src.transform,
                lr["pixel_width"],
                hr["pixel_width"],
            )

            if not transform_ok:
                pair_errors.extend(transform_errors)

            # Compare geographic bounds.
            lr_bounds = lr_src.bounds
            hr_bounds = hr_src.bounds

            for label, a, b in (
                ("left", lr_bounds.left, hr_bounds.left),
                ("right", lr_bounds.right, hr_bounds.right),
                ("top", lr_bounds.top, hr_bounds.top),
                ("bottom", lr_bounds.bottom, hr_bounds.bottom),
            ):
                if not nearly_equal(a, b):
                    pair_errors.append(
                        f"bounds {label} mismatch: "
                        f"LR={a}, HR={b}"
                    )

    # Combine structural errors.
    for error in lr["errors"]:
        pair_errors.append(f"LR: {error}")

    for error in hr["errors"]:
        pair_errors.append(f"HR: {error}")

    for warning in lr["warnings"]:
        pair_warnings.append(f"LR: {warning}")

    for warning in hr["warnings"]:
        pair_warnings.append(f"HR: {warning}")

    result = {
        "filename": filename,
        "lr_path": relative_path(lr_path, data_root),
        "hr_path": relative_path(hr_path, data_root),

        "lr_readable": lr["readable"],
        "hr_readable": hr["readable"],

        "lr_width": lr["width"],
        "lr_height": lr["height"],
        "lr_bands": lr["bands"],
        "lr_dtype": lr["dtype"],
        "lr_crs": lr["crs"],
        "lr_pixel_width": lr["pixel_width"],
        "lr_pixel_height": lr["pixel_height"],
        "lr_nodata": lr["nodata"],
        "lr_nodata_percent": lr["nodata_percent"],
        "lr_min": lr["min_value"],
        "lr_max": lr["max_value"],

        "hr_width": hr["width"],
        "hr_height": hr["height"],
        "hr_bands": hr["bands"],
        "hr_dtype": hr["dtype"],
        "hr_crs": hr["crs"],
        "hr_pixel_width": hr["pixel_width"],
        "hr_pixel_height": hr["pixel_height"],
        "hr_nodata": hr["nodata"],
        "hr_nodata_percent": hr["nodata_percent"],
        "hr_min": hr["min_value"],
        "hr_max": hr["max_value"],

        "errors": " | ".join(pair_errors),
        "warnings": " | ".join(pair_warnings),

        "status": (
            "PASS"
            if not pair_errors
            else "FAIL"
        ),
    }

    return result, {
        "lr": lr,
        "hr": hr,
        "errors": pair_errors,
        "warnings": pair_warnings,
    }


# ============================================================
# REPORTING
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_pair_result(index: int, total: int, row: dict[str, Any]) -> None:

    status = row["status"]

    symbol = "PASS" if status == "PASS" else "FAIL"

    print(
        f"[{index:4d}/{total:4d}] "
        f"{symbol:<4} "
        f"{row['filename']}"
    )

    if row["errors"]:
        print(
            f"             ERRORS: {row['errors']}"
        )

    if row["warnings"]:
        print(
            f"             WARNINGS: {row['warnings']}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    args = parse_args()

    data_root = Path(args.data_root).resolve()

    lr_dir = data_root / "s2_l2a_10m"
    hr_dir = data_root / "neon_2.5m_linearized"

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else data_root / "verification"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("SEN2NEON DATASET VERIFICATION")
    print("=" * 72)
    print(f"Data root : {data_root}")
    print(f"LR folder : {lr_dir}")
    print(f"HR folder : {hr_dir}")
    print()

    # --------------------------------------------------------
    # Validate directories.
    # --------------------------------------------------------

    if not data_root.exists():
        print(
            f"ERROR: Data root does not exist:\n{data_root}"
        )
        return 2

    if not lr_dir.exists():
        print(
            f"ERROR: LR directory does not exist:\n{lr_dir}"
        )
        return 2

    if not hr_dir.exists():
        print(
            f"ERROR: HR directory does not exist:\n{hr_dir}"
        )
        return 2

    # --------------------------------------------------------
    # Find TIFF files.
    # --------------------------------------------------------

    print("Scanning TIFF files...")

    lr_files = list_tiffs(lr_dir)
    hr_files = list_tiffs(hr_dir)

    print(f"LR TIFF files : {len(lr_files)}")
    print(f"HR TIFF files : {len(hr_files)}")
    print()

    lr_names = set(lr_files)
    hr_names = set(hr_files)

    missing_hr = sorted(lr_names - hr_names)
    missing_lr = sorted(hr_names - lr_names)

    common_names = sorted(lr_names & hr_names)

    if args.max_files is not None:
        if args.max_files <= 0:
            print("ERROR: --max-files must be > 0")
            return 2

        common_names = common_names[:args.max_files]

    print(f"Matched pairs : {len(common_names)}")
    print(f"Missing HR    : {len(missing_hr)}")
    print(f"Missing LR    : {len(missing_lr)}")
    print()

    # --------------------------------------------------------
    # Pair verification.
    # --------------------------------------------------------

    rows: list[dict[str, Any]] = []

    total = len(common_names)

    for index, filename in enumerate(common_names, start=1):

        row, _ = check_pair(
            filename=filename,
            lr_path=lr_files[filename],
            hr_path=hr_files[filename],
            data_root=data_root,
            skip_value_check=args.skip_value_check,
            sample_step=max(1, args.sample_step),
        )

        rows.append(row)

        print_pair_result(
            index,
            total,
            row,
        )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    passed = sum(
        1
        for row in rows
        if row["status"] == "PASS"
    )

    failed = len(rows) - passed

    summary = {
        "dataset": "SEN2NEON",
        "data_root": str(data_root),

        "expected": {
            "bands": EXPECTED_BANDS,
            "lr_shape": [
                EXPECTED_BANDS,
                LR_HEIGHT,
                LR_WIDTH,
            ],
            "hr_shape": [
                EXPECTED_BANDS,
                HR_HEIGHT,
                HR_WIDTH,
            ],
            "lr_resolution_m": LR_RESOLUTION,
            "hr_resolution_m": HR_RESOLUTION,
            "scale_factor": EXPECTED_SCALE,
            "lr_dtype": EXPECTED_LR_DTYPE,
            "hr_dtype": EXPECTED_HR_DTYPE,
            "lr_nodata": EXPECTED_LR_NODATA,
            "hr_nodata": EXPECTED_HR_NODATA,
            "reflectance_scale": REFLECTANCE_SCALE,
        },

        "files": {
            "lr_files": len(lr_files),
            "hr_files": len(hr_files),
            "matched_pairs": len(common_names),
            "missing_hr": len(missing_hr),
            "missing_lr": len(missing_lr),
        },

        "verification": {
            "passed": passed,
            "failed": failed,
            "status": (
                "READY"
                if (
                    failed == 0
                    and len(missing_hr) == 0
                    and len(missing_lr) == 0
                    and len(common_names) > 0
                )
                else "NOT_READY"
            ),
        },

        "missing_hr": missing_hr,
        "missing_lr": missing_lr,
    }

    # --------------------------------------------------------
    # Write reports.
    # --------------------------------------------------------

    csv_path = output_dir / "verification_report.csv"
    json_path = output_dir / "verification_summary.json"

    write_csv(
        csv_path,
        rows,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console report.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)

    print(f"LR files           : {len(lr_files)}")
    print(f"HR files           : {len(hr_files)}")
    print(f"Matched pairs      : {len(common_names)}")
    print(f"Missing HR         : {len(missing_hr)}")
    print(f"Missing LR         : {len(missing_lr)}")
    print(f"Pairs passed       : {passed}")
    print(f"Pairs failed       : {failed}")

    print()
    print("Expected LR        : 12 × 256 × 256 @ 10 m")
    print("Expected HR        : 12 × 1024 × 1024 @ 2.5 m")
    print("Expected scale     : 4×")
    print("Expected LR dtype  : uint16")
    print("Expected HR dtype  : uint16")
    print("Expected LR nodata : 65535")
    print("Expected HR nodata : 0")

    print()
    print(f"CSV report         : {csv_path}")
    print(f"JSON summary       : {json_path}")

    status = summary["verification"]["status"]

    print()
    print("-" * 72)

    if status == "READY":
        print("STATUS: READY FOR NEXT PREPROCESSING STEP")
        print("-" * 72)
        print()
        return 0

    print("STATUS: NOT READY")
    print("-" * 72)
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())