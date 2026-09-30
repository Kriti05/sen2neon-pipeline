#!/usr/bin/env python3
"""
SEN2NEON Train / Validation / Test Split Generator
===================================================

Creates leakage-safe scene/acquisition-level splits.

IMPORTANT:
-----------
Splitting happens at GROUP level, NOT individual patch level.

Preferred grouping:
    neon_acquisition_id

Fallback grouping:
    filename prefix before "__"

Example:
    2018_MLBS_3__0_2.tif
    2018_MLBS_3__1_1.tif
    2018_MLBS_3__1_2.tif

Fallback group:
    2018_MLBS_3

This prevents tiles belonging to the same acquisition from
being randomly distributed across train/validation/test.

Expected data:

DATA_ROOT/
├── s2_l2a_10m/
├── neon_2.5m_linearized/
└── metadata.csv                  optional

Output:

DATA_ROOT/
└── splits/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── splits.json

Usage:

    python preprocessing/create_splits.py \
        --data-root SEN2NEON_5GB

Optional:

    python preprocessing/create_splits.py \
        --data-root SEN2NEON_5GB \
        --train-ratio 0.70 \
        --val-ratio 0.15 \
        --test-ratio 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

DEFAULT_SEED = 42

LR_DIR_NAME = "s2_l2a_10m"
HR_DIR_NAME = "neon_2.5m_linearized"

SPLIT_DIR_NAME = "splits"

TIFF_EXTENSIONS = {".tif", ".tiff"}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-safe SEN2NEON "
            "train/validation/test splits."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="SEN2NEON_5GB",
        help=(
            "Root directory containing "
            "s2_l2a_10m and neon_2.5m_linearized."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Optional metadata.csv path. "
            "If omitted, <data-root>/metadata.csv is used "
            "when present."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. "
            "Default: <data-root>/splits"
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Training ratio. Default: 0.70",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation ratio. Default: 0.15",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test ratio. Default: 0.15",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Optional limit for testing. "
            "Example: --max-pairs 50"
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:

    ratios = [
        train_ratio,
        val_ratio,
        test_ratio,
    ]

    if any(r < 0 for r in ratios):
        raise ValueError(
            "Split ratios cannot be negative."
        )

    total = (
        train_ratio
        + val_ratio
        + test_ratio
    )

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"Split ratios must sum to 1.0. "
            f"Got {total:.8f}"
        )

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be greater than zero."
        )

    if val_ratio <= 0:
        raise ValueError(
            "val_ratio must be greater than zero."
        )

    if test_ratio <= 0:
        raise ValueError(
            "test_ratio must be greater than zero."
        )


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_tiffs(directory: Path) -> dict[str, Path]:

    if not directory.exists():
        return {}

    result: dict[str, Path] = {}

    for path in directory.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in TIFF_EXTENSIONS
        ):
            result[path.name] = path

    return result


def find_pairs(
    data_root: Path,
    max_pairs: int | None = None,
) -> pd.DataFrame:

    lr_dir = data_root / LR_DIR_NAME
    hr_dir = data_root / HR_DIR_NAME

    lr_files = find_tiffs(lr_dir)
    hr_files = find_tiffs(hr_dir)

    common = sorted(
        set(lr_files.keys())
        & set(hr_files.keys())
    )

    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError(
                "--max-pairs must be greater than zero."
            )

        common = common[:max_pairs]

    rows = []

    for filename in common:

        rows.append(
            {
                "name": filename,
                "lr_path": str(
                    Path(LR_DIR_NAME) / filename
                ),
                "hr_path": str(
                    Path(HR_DIR_NAME) / filename
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FALLBACK GROUP EXTRACTION
# ============================================================

def infer_group_from_filename(
    filename: str,
) -> str:
    """
    Example:

        2018_MLBS_3__0_2.tif
                ^^^^^^^^^

    returns:

        2018_MLBS_3

    The SEN2NEON filenames use the double underscore
    to separate the acquisition/site portion from the
    tile index.
    """

    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__", 1)[0]

    # Conservative fallback.
    #
    # If no "__" exists, remove the final _number_number
    # pattern when present.
    match = re.match(
        r"^(.*?)(?:_\d+_\d+)$",
        stem,
    )

    if match:
        return match.group(1)

    return stem


# ============================================================
# METADATA
# ============================================================

def find_metadata(
    data_root: Path,
    metadata_argument: str | None,
) -> Path | None:

    if metadata_argument:

        path = Path(metadata_argument).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file does not exist:\n{path}"
            )

        return path

    candidates = [
        data_root / "metadata.csv",
        data_root / "metadata.parquet",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return None


def load_metadata(
    metadata_path: Path | None,
) -> pd.DataFrame | None:

    if metadata_path is None:
        print(
            "Metadata file not found."
        )
        print(
            "Using filename-derived acquisition groups."
        )
        return None

    print(
        f"Reading metadata: {metadata_path}"
    )

    suffix = metadata_path.suffix.lower()

    if suffix == ".csv":

        metadata = pd.read_csv(
            metadata_path,
        )

    elif suffix == ".parquet":

        try:
            metadata = pd.read_parquet(
                metadata_path,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow or fastparquet.\n"
                "Install with:\n"
                "    pip install pyarrow"
            ) from exc

    else:

        raise ValueError(
            "Metadata must be CSV or Parquet."
        )

    if metadata.empty:
        raise ValueError(
            "Metadata file is empty."
        )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    return metadata


# ============================================================
# METADATA MERGING
# ============================================================

def normalize_name(value: Any) -> str:

    if pd.isna(value):
        return ""

    return Path(str(value)).name


def attach_metadata(
    pairs: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:

    result = pairs.copy()

    # --------------------------------------------------------
    # No metadata.
    # --------------------------------------------------------

    if metadata is None:

        result["id"] = (
            result["name"]
            .map(lambda x: Path(x).stem)
        )

        result["group_id"] = (
            result["name"]
            .map(infer_group_from_filename)
        )

        result["metadata_available"] = False

        return result

    # --------------------------------------------------------
    # Identify filename column.
    # --------------------------------------------------------

    metadata = metadata.copy()

    filename_column = None

    for candidate in (
        "name",
        "filename",
        "file_name",
    ):

        if candidate in metadata.columns:

            filename_column = candidate
            break

    if filename_column is None:

        # Try path columns.
        for candidate in (
            "lr",
            "hr",
            "hr_2_5m_path",
        ):

            if candidate in metadata.columns:

                filename_column = candidate
                break

    if filename_column is None:

        raise ValueError(
            "Could not find a filename/path column in metadata."
        )

    metadata["_filename_key"] = (
        metadata[filename_column]
        .map(normalize_name)
    )

    # Remove duplicate metadata records.
    metadata = metadata.drop_duplicates(
        subset=["_filename_key"],
        keep="first",
    )

    result = result.merge(
        metadata,
        how="left",
        left_on="name",
        right_on="_filename_key",
        suffixes=("", "_metadata"),
    )

    # --------------------------------------------------------
    # Acquisition grouping.
    # --------------------------------------------------------

    group_column = None

    for candidate in (
        "neon_acquisition_id",
        "neon_asset_id",
    ):

        if candidate in result.columns:

            valid_count = (
                result[candidate]
                .notna()
                .sum()
            )

            if valid_count > 0:
                group_column = candidate
                break

    if group_column is not None:

        result["group_id"] = (
            result[group_column]
            .fillna(
                result["name"].map(
                    infer_group_from_filename
                )
            )
            .astype(str)
        )

        result["group_source"] = group_column

    else:

        result["group_id"] = (
            result["name"]
            .map(infer_group_from_filename)
        )

        result["group_source"] = (
            "filename_fallback"
        )

    result["metadata_available"] = (
        result["_filename_key"].notna()
    )

    # ID.
    if "id" not in result.columns:

        result["id"] = (
            result["name"]
            .map(lambda x: Path(x).stem)
        )

    return result


# ============================================================
# GROUP SPLITTING
# ============================================================

def create_group_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:

    df = dataframe.copy()

    groups = sorted(
        df["group_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(groups) < 3:

        raise RuntimeError(
            "At least 3 unique acquisition groups are required "
            "to create train/validation/test splits."
        )

    rng = random.Random(seed)

    # Deterministic randomization.
    groups = list(groups)
    rng.shuffle(groups)

    # --------------------------------------------------------
    # Assign whole groups.
    #
    # We target group counts rather than individual tiles.
    # This guarantees no acquisition leakage.
    # --------------------------------------------------------

    total_groups = len(groups)

    target_train = max(
        1,
        round(total_groups * train_ratio),
    )

    target_val = max(
        1,
        round(total_groups * val_ratio),
    )

    # Ensure at least one test group.
    target_test = (
        total_groups
        - target_train
        - target_val
    )

    # Correct rounding if necessary.
    while target_test < 1:

        if target_train > target_val and target_train > 1:
            target_train -= 1
        elif target_val > 1:
            target_val -= 1
        else:
            break

        target_test = (
            total_groups
            - target_train
            - target_val
        )

    # If rounding caused too many groups.
    while (
        target_train
        + target_val
        + target_test
        > total_groups
    ):

        if target_train > 1:
            target_train -= 1
        elif target_val > 1:
            target_val -= 1
        else:
            target_test -= 1

    train_groups = set(
        groups[:target_train]
    )

    val_start = target_train

    val_groups = set(
        groups[
            val_start:
            val_start + target_val
        ]
    )

    test_start = (
        target_train
        + target_val
    )

    test_groups = set(
        groups[test_start:]
    )

    # --------------------------------------------------------
    # Safety check.
    # --------------------------------------------------------

    if not train_groups:
        raise RuntimeError(
            "Training split received zero groups."
        )

    if not val_groups:
        raise RuntimeError(
            "Validation split received zero groups."
        )

    if not test_groups:
        raise RuntimeError(
            "Test split received zero groups."
        )

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # Assign split labels.
    # --------------------------------------------------------

    def assign(group: str) -> str:

        if group in train_groups:
            return "train"

        if group in val_groups:
            return "val"

        if group in test_groups:
            return "test"

        raise RuntimeError(
            f"Unknown group: {group}"
        )

    df["split"] = (
        df["group_id"]
        .astype(str)
        .map(assign)
    )

    return df


# ============================================================
# STATISTICS
# ============================================================

def print_split_statistics(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print("SPLIT STATISTICS")
    print("=" * 72)

    total = len(df)

    for split in (
        "train",
        "val",
        "test",
    ):

        subset = df[
            df["split"] == split
        ]

        count = len(subset)

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        groups = (
            subset["group_id"]
            .nunique()
        )

        print(
            f"{split.upper():5s} "
            f"pairs={count:4d} "
            f"({percentage:6.2f}%) "
            f"groups={groups:4d}"
        )

    # --------------------------------------------------------
    # Group leakage check.
    # --------------------------------------------------------

    group_sets = {
        split: set(
            df.loc[
                df["split"] == split,
                "group_id",
            ].astype(str)
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    leakage = (
        (group_sets["train"] & group_sets["val"])
        | (group_sets["train"] & group_sets["test"])
        | (group_sets["val"] & group_sets["test"])
    )

    print()
    print(
        f"Group leakage : "
        f"{len(leakage)}"
    )

    if leakage:
        print(
            "ERROR: group leakage detected:"
        )

        for group in sorted(leakage):
            print(
                f"  - {group}"
            )

    else:
        print(
            "Group leakage : NONE"
        )


def print_landcover_statistics(
    df: pd.DataFrame,
) -> None:

    possible_columns = [
        "land_cover_superclass",
        "LC_superclass_text",
        "land_cover",
        "LC_detail_text",
    ]

    column = None

    for candidate in possible_columns:

        if candidate in df.columns:

            if df[candidate].notna().any():
                column = candidate
                break

    if column is None:
        return

    print()
    print("=" * 72)
    print(
        f"LAND-COVER DISTRIBUTION ({column})"
    )
    print("=" * 72)

    for split in (
        "train",
        "val",
        "test",
    ):

        subset = df[
            df["split"] == split
        ]

        counts = (
            subset[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        print()
        print(f"{split.upper()}:")

        for label, count in counts.items():

            pct = (
                100.0 * count / len(subset)
                if len(subset)
                else 0.0
            )

            print(
                f"  {label:<30} "
                f"{count:4d} "
                f"({pct:5.1f}%)"
            )


# ============================================================
# OUTPUT
# ============================================================

def write_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Select useful columns first.
    # --------------------------------------------------------

    preferred_columns = [
        "id",
        "name",
        "lr_path",
        "hr_path",
        "group_id",
        "group_source",
        "split",
        "lon",
        "lat",
        "centroid_lon",
        "centroid_lat",
        "land_cover",
        "land_cover_detail",
        "land_cover_superclass",
        "LC_detail_text",
        "LC_superclass_text",
        "neon_acquisition_id",
        "neon_asset_id",
        "neon_date",
        "s2_date",
        "cloud_score_plus_cdf",
        "s2_nodata_percent",
        "neon_nodata_percent",
    ]

    selected_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    # Add anything else from the original dataframe
    # after the important columns.
    remaining_columns = [
        column
        for column in df.columns
        if column not in selected_columns
        and not column.startswith("_")
    ]

    final_columns = (
        selected_columns
        + remaining_columns
    )

    output_df = df[
        final_columns
    ].copy()

    # --------------------------------------------------------
    # Write individual split files.
    # --------------------------------------------------------

    for split in (
        "train",
        "val",
        "test",
    ):

        split_df = output_df[
            output_df["split"] == split
        ].copy()

        split_df.to_csv(
            output_dir / f"{split}.csv",
            index=False,
        )

    # --------------------------------------------------------
    # JSON summary.
    # --------------------------------------------------------

    groups_by_split = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        groups_by_split[split] = sorted(
            output_df.loc[
                output_df["split"] == split,
                "group_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

    summary = {
        "dataset": "SEN2NEON",
        "seed": args.seed,

        "ratios_requested": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },

        "pairs": {
            "total": int(len(output_df)),
            "train": int(
                (output_df["split"] == "train").sum()
            ),
            "val": int(
                (output_df["split"] == "val").sum()
            ),
            "test": int(
                (output_df["split"] == "test").sum()
            ),
        },

        "groups": {
            "total": int(
                output_df["group_id"].nunique()
            ),
            "train": len(groups_by_split["train"]),
            "val": len(groups_by_split["val"]),
            "test": len(groups_by_split["test"]),
        },

        "groups_by_split": groups_by_split,

        "grouping": (
            "neon_acquisition_id"
            if "neon_acquisition_id" in output_df.columns
            and output_df["group_source"].eq(
                "neon_acquisition_id"
            ).any()
            else "filename_fallback"
        ),

        "leakage_check": {
            "train_val": len(
                set(groups_by_split["train"])
                & set(groups_by_split["val"])
            ),
            "train_test": len(
                set(groups_by_split["train"])
                & set(groups_by_split["test"])
            ),
            "val_test": len(
                set(groups_by_split["val"])
                & set(groups_by_split["test"])
            ),
        },
    }

    with (
        output_dir / "splits.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Master CSV.
    # --------------------------------------------------------

    output_df.to_csv(
        output_dir / "all_pairs_with_splits.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        output_dir / "train.csv"
    )

    print(
        output_dir / "val.csv"
    )

    print(
        output_dir / "test.csv"
    )

    print(
        output_dir / "all_pairs_with_splits.csv"
    )

    print(
        output_dir / "splits.json"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_output(
    df: pd.DataFrame,
) -> None:

    required_columns = {
        "name",
        "lr_path",
        "hr_path",
        "group_id",
        "split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Output missing required columns: {missing}"
        )

    # Every pair must have exactly one split.
    if df["split"].isna().any():
        raise RuntimeError(
            "Some pairs do not have a split."
        )

    allowed = {
        "train",
        "val",
        "test",
    }

    invalid = (
        set(df["split"].unique())
        - allowed
    )

    if invalid:
        raise RuntimeError(
            f"Invalid split labels: {invalid}"
        )

    # Check duplicated filenames.
    duplicates = df[
        df["name"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:
        raise RuntimeError(
            "Duplicate filenames found in split dataset."
        )

    # Group leakage.
    split_groups = {
        split: set(
            df.loc[
                df["split"] == split,
                "group_id",
            ].astype(str)
        )
        for split in allowed
    }

    for first, second in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):

        overlap = (
            split_groups[first]
            & split_groups[second]
        )

        if overlap:
            raise RuntimeError(
                f"Group leakage between "
                f"{first} and {second}: "
                f"{sorted(overlap)}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    args = parse_args()

    try:
        validate_ratios(
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}"
        )

        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(
        args.data_root
    ).resolve()

    if not data_root.exists():

        print(
            f"ERROR: data root does not exist:\n"
            f"{data_root}"
        )

        return 2

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else data_root / SPLIT_DIR_NAME
    )

    print()
    print("=" * 72)
    print("SEN2NEON SPLIT GENERATION")
    print("=" * 72)

    print(
        f"Data root     : {data_root}"
    )

    print(
        f"Output        : {output_dir}"
    )

    print(
        f"Train ratio   : {args.train_ratio:.2f}"
    )

    print(
        f"Val ratio     : {args.val_ratio:.2f}"
    )

    print(
        f"Test ratio    : {args.test_ratio:.2f}"
    )

    print(
        f"Random seed   : {args.seed}"
    )

    print()

    # --------------------------------------------------------
    # Find paired TIFFs.
    # --------------------------------------------------------

    print(
        "Finding LR/HR pairs..."
    )

    pairs = find_pairs(
        data_root,
        args.max_pairs,
    )

    if pairs.empty:

        print(
            "ERROR: no matched LR/HR pairs found."
        )

        print(
            f"Expected:\n"
            f"  {data_root / LR_DIR_NAME}\n"
            f"  {data_root / HR_DIR_NAME}"
        )

        return 2

    print(
        f"Matched pairs: {len(pairs)}"
    )

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    metadata_path = find_metadata(
        data_root,
        args.metadata,
    )

    metadata = load_metadata(
        metadata_path,
    )

    # --------------------------------------------------------
    # Attach metadata/group IDs.
    # --------------------------------------------------------

    pairs = attach_metadata(
        pairs,
        metadata,
    )

    print()

    if (
        "group_source" in pairs.columns
        and pairs["group_source"].eq(
            "neon_acquisition_id"
        ).any()
    ):

        print(
            "Grouping method: "
            "NEON acquisition ID"
        )

    else:

        print(
            "Grouping method: "
            "filename-derived acquisition group"
        )

        print(
            "WARNING: For final experiments, "
            "download metadata.csv and rerun."
        )

    print(
        f"Unique groups: "
        f"{pairs['group_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Create splits.
    # --------------------------------------------------------

    print()
    print(
        "Creating acquisition-level splits..."
    )

    pairs = create_group_split(
        dataframe=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_output(
        pairs,
    )

    # --------------------------------------------------------
    # Print stats.
    # --------------------------------------------------------

    print_split_statistics(
        pairs,
    )

    print_landcover_statistics(
        pairs,
    )

    # --------------------------------------------------------
    # Write.
    # --------------------------------------------------------

    write_outputs(
        pairs,
        output_dir,
        args,
    )

    # --------------------------------------------------------
    # Final.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SPLIT GENERATION COMPLETE")
    print("=" * 72)

    print(
        "STATUS: READY FOR PATCH GENERATION"
    )

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())