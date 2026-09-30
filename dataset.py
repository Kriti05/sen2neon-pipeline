#!/usr/bin/env python3
"""
SEN2NEON PyTorch Dataset
========================

Reads aligned LR/HR windows directly from SEN2NEON GeoTIFFs.

Input:
    LR  : 12 x 64 x 64
    HR  : 12 x 256 x 256

Scale:
    4x

Storage:
    uint16

Normalization:
    uint16 / 10000.0

Nodata:
    LR = 65535
    HR = 0

Output:
    lr            [12, 64, 64]    float32
    hr            [12, 256, 256]  float32
    lr_mask       [12, 64, 64]    bool
    hr_mask       [12, 256, 256]  bool
    metadata      dict

The Dataset intentionally reads windows lazily so the original
GeoTIFFs are not duplicated on disk.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window


# ============================================================
# CONSTANTS
# ============================================================

NUM_BANDS = 12

REFLECTANCE_SCALE = 10000.0

LR_NODATA = 65535
HR_NODATA = 0

EXPECTED_LR_SIZE = 64
EXPECTED_HR_SIZE = 256

EXPECTED_SCALE = 4

BAND_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_reflectance(
    array: np.ndarray,
    nodata: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert uint16 reflectance to float32 [approximately 0,1].

    Returns:
        normalized
        valid_mask
    """

    if array.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 data, got {array.dtype}"
        )

    valid_mask = array != nodata

    normalized = (
        array.astype(np.float32)
        / REFLECTANCE_SCALE
    )

    normalized[~valid_mask] = 0.0

    # Keep the physical range non-negative.
    normalized = np.maximum(
        normalized,
        0.0,
    )

    return normalized, valid_mask


def numpy_to_tensor(
    array: np.ndarray,
) -> torch.Tensor:
    """
    Convert [C,H,W] numpy array to torch float32 tensor.
    """

    return torch.from_numpy(
        np.ascontiguousarray(array)
    ).float()


# ============================================================
# AUGMENTATION
# ============================================================

def apply_augmentation(
    lr: np.ndarray,
    hr: np.ndarray,
    lr_mask: np.ndarray,
    hr_mask: np.ndarray,
    training: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply identical spatial transformations to LR and HR.

    IMPORTANT:
        The exact same transform is applied to LR and HR.

    Allowed transformations:
        - horizontal flip
        - vertical flip
        - 90-degree rotation
    """

    if not training:
        return (
            lr,
            hr,
            lr_mask,
            hr_mask,
        )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        lr = np.flip(
            lr,
            axis=2,
        )

        hr = np.flip(
            hr,
            axis=2,
        )

        lr_mask = np.flip(
            lr_mask,
            axis=2,
        )

        hr_mask = np.flip(
            hr_mask,
            axis=2,
        )

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        lr = np.flip(
            lr,
            axis=1,
        )

        hr = np.flip(
            hr,
            axis=1,
        )

        lr_mask = np.flip(
            lr_mask,
            axis=1,
        )

        hr_mask = np.flip(
            hr_mask,
            axis=1,
        )

    # --------------------------------------------------------
    # 90-degree rotations
    # --------------------------------------------------------

    k = random.randint(
        0,
        3,
    )

    if k != 0:

        lr = np.rot90(
            lr,
            k=k,
            axes=(1, 2),
        )

        hr = np.rot90(
            hr,
            k=k,
            axes=(1, 2),
        )

        lr_mask = np.rot90(
            lr_mask,
            k=k,
            axes=(1, 2),
        )

        hr_mask = np.rot90(
            hr_mask,
            k=k,
            axes=(1, 2),
        )

    # --------------------------------------------------------
    # np.flip/np.rot90 can create negative strides.
    # Make contiguous arrays before converting to Torch.
    # --------------------------------------------------------

    lr = np.ascontiguousarray(lr)
    hr = np.ascontiguousarray(hr)

    lr_mask = np.ascontiguousarray(
        lr_mask
    )

    hr_mask = np.ascontiguousarray(
        hr_mask
    )

    return (
        lr,
        hr,
        lr_mask,
        hr_mask,
    )


# ============================================================
# DATASET
# ============================================================

class SEN2NEONDataset(Dataset):
    """
    PyTorch Dataset for SEN2NEON.

    Parameters
    ----------
    patch_index:
        CSV containing patch records.

    data_root:
        SEN2NEON root directory.

    split:
        train / val / test.

    training:
        Enables random spatial augmentation.

    return_masks:
        Whether to return LR/HR validity masks.

    max_samples:
        Optional debugging limit.
    """

    def __init__(
        self,
        patch_index: str | Path,
        data_root: str | Path,
        split: str,
        training: bool = False,
        return_masks: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.patch_index = Path(
            patch_index
        )

        self.data_root = Path(
            data_root
        )

        self.split = split

        self.training = training

        self.return_masks = return_masks

        # ----------------------------------------------------
        # Load index
        # ----------------------------------------------------

        if not self.patch_index.exists():
            raise FileNotFoundError(
                f"Patch index not found: "
                f"{self.patch_index}"
            )

        df = pd.read_csv(
            self.patch_index
        )

        if "split" not in df.columns:
            raise ValueError(
                "Patch index must contain "
                "a 'split' column."
            )

        df = df[
            df["split"] == split
        ].reset_index(
            drop=True
        )

        if max_samples is not None:
            df = df.head(
                max_samples
            ).copy()

        if df.empty:
            raise ValueError(
                f"No patches found for split "
                f"'{split}'."
            )

        self.df = df

        # ----------------------------------------------------
        # GeoTIFF handles are opened lazily per worker.
        #
        # This is important for DataLoader multiprocessing.
        # ----------------------------------------------------

        self._lr_handles: Dict[
            str,
            rasterio.io.DatasetReader,
        ] = {}

        self._hr_handles: Dict[
            str,
            rasterio.io.DatasetReader,
        ] = {}

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self) -> int:
        return len(self.df)

    # ========================================================
    # PATH
    # ========================================================

    def _resolve_path(
        self,
        path_value: str,
    ) -> Path:

        path = Path(
            str(path_value)
        )

        if path.is_absolute():
            return path

        return self.data_root / path

    # ========================================================
    # RASTER HANDLES
    # ========================================================

    def _get_lr_handle(
        self,
        path: Path,
    ):

        key = str(
            path.resolve()
        )

        if key not in self._lr_handles:

            if not path.exists():
                raise FileNotFoundError(
                    f"LR file does not exist: "
                    f"{path}"
                )

            self._lr_handles[key] = (
                rasterio.open(path)
            )

        return self._lr_handles[key]

    def _get_hr_handle(
        self,
        path: Path,
    ):

        key = str(
            path.resolve()
        )

        if key not in self._hr_handles:

            if not path.exists():
                raise FileNotFoundError(
                    f"HR file does not exist: "
                    f"{path}"
                )

            self._hr_handles[key] = (
                rasterio.open(path)
            )

        return self._hr_handles[key]

    # ========================================================
    # READ WINDOW
    # ========================================================

    def _read_lr(
        self,
        row: pd.Series,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        path = self._resolve_path(
            row["lr_path"]
        )

        x = int(
            row["lr_x"]
        )

        y = int(
            row["lr_y"]
        )

        size = int(
            row["lr_size"]
        )

        window = Window(
            col_off=x,
            row_off=y,
            width=size,
            height=size,
        )

        src = self._get_lr_handle(
            path
        )

        array = src.read(
            window=window
        )

        normalized, mask = (
            normalize_reflectance(
                array,
                LR_NODATA,
            )
        )

        return normalized, mask

    def _read_hr(
        self,
        row: pd.Series,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        path = self._resolve_path(
            row["hr_path"]
        )

        x = int(
            row["hr_x"]
        )

        y = int(
            row["hr_y"]
        )

        size = int(
            row["hr_size"]
        )

        window = Window(
            col_off=x,
            row_off=y,
            width=size,
            height=size,
        )

        src = self._get_hr_handle(
            path
        )

        array = src.read(
            window=window
        )

        normalized, mask = (
            normalize_reflectance(
                array,
                HR_NODATA,
            )
        )

        return normalized, mask

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        row = self.df.iloc[
            index
        ]

        lr, lr_mask = (
            self._read_lr(row)
        )

        hr, hr_mask = (
            self._read_hr(row)
        )

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        expected_lr = (
            NUM_BANDS,
            int(row["lr_size"]),
            int(row["lr_size"]),
        )

        expected_hr = (
            NUM_BANDS,
            int(row["hr_size"]),
            int(row["hr_size"]),
        )

        if lr.shape != expected_lr:
            raise RuntimeError(
                f"Unexpected LR shape "
                f"{lr.shape}; expected "
                f"{expected_lr}"
            )

        if hr.shape != expected_hr:
            raise RuntimeError(
                f"Unexpected HR shape "
                f"{hr.shape}; expected "
                f"{expected_hr}"
            )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        (
            lr,
            hr,
            lr_mask,
            hr_mask,
        ) = apply_augmentation(
            lr=lr,
            hr=hr,
            lr_mask=lr_mask,
            hr_mask=hr_mask,
            training=self.training,
        )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        lr_tensor = numpy_to_tensor(
            lr
        )

        hr_tensor = numpy_to_tensor(
            hr
        )

        result = {
            "lr": lr_tensor,
            "hr": hr_tensor,
        }

        # ----------------------------------------------------
        # Masks
        # ----------------------------------------------------

        if self.return_masks:

            result["lr_mask"] = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        lr_mask
                    )
                ).bool()
            )

            result["hr_mask"] = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        hr_mask
                    )
                ).bool()
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        result["metadata"] = {
            "patch_id": str(
                row.get(
                    "patch_id",
                    index,
                )
            ),

            "split": str(
                row.get(
                    "split",
                    self.split,
                )
            ),

            "lr_path": str(
                row["lr_path"]
            ),

            "hr_path": str(
                row["hr_path"]
            ),

            "lr_x": int(
                row["lr_x"]
            ),

            "lr_y": int(
                row["lr_y"]
            ),

            "hr_x": int(
                row["hr_x"]
            ),

            "hr_y": int(
                row["hr_y"]
            ),

            "scale": int(
                row.get(
                    "scale",
                    EXPECTED_SCALE,
                )
            ),
        }

        # ----------------------------------------------------
        # Preserve useful optional metadata.
        # ----------------------------------------------------

        optional_metadata = [
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

        for key in optional_metadata:

            if key in row.index:

                value = row[key]

                if pd.isna(value):
                    value = ""

                result["metadata"][key] = (
                    value
                )

        return result

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self) -> None:

        for handle in (
            self._lr_handles.values()
        ):
            try:
                handle.close()
            except Exception:
                pass

        for handle in (
            self._hr_handles.values()
        ):
            try:
                handle.close()
            except Exception:
                pass

        self._lr_handles.clear()
        self._hr_handles.clear()

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_sen2neon(
    batch,
) -> Dict:

    lr = torch.stack(
        [
            item["lr"]
            for item in batch
        ],
        dim=0,
    )

    hr = torch.stack(
        [
            item["hr"]
            for item in batch
        ],
        dim=0,
    )

    result = {
        "lr": lr,
        "hr": hr,
        "metadata": [
            item["metadata"]
            for item in batch
        ],
    }

    if "lr_mask" in batch[0]:

        result["lr_mask"] = (
            torch.stack(
                [
                    item["lr_mask"]
                    for item in batch
                ],
                dim=0,
            )
        )

    if "hr_mask" in batch[0]:

        result["hr_mask"] = (
            torch.stack(
                [
                    item["hr_mask"]
                    for item in batch
                ],
                dim=0,
            )
        )

    return result


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def worker_init_fn(
    worker_id: int,
) -> None:

    # Give each worker a different random seed.
    seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloader(
    patch_index: str | Path,
    data_root: str | Path,
    split: str,
    batch_size: int = 4,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    training: Optional[bool] = None,
    return_masks: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Convenience function for training/evaluation.
    """

    if training is None:
        training = (
            split == "train"
        )

    if shuffle is None:
        shuffle = training

    dataset = SEN2NEONDataset(
        patch_index=patch_index,
        data_root=data_root,
        split=split,
        training=training,
        return_masks=return_masks,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(
            pin_memory
            and torch.cuda.is_available()
        ),
        drop_last=drop_last,
        collate_fn=collate_sen2neon,
        worker_init_fn=worker_init_fn,
        persistent_workers=(
            num_workers > 0
        ),
    )

    return loader


# ============================================================
# SELF TEST
# ============================================================

def run_self_test(
    data_root: Path,
    patch_index: Path,
    max_samples: int = 2,
) -> None:

    print()
    print("=" * 72)
    print("SEN2NEON DATASET SELF-TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = SEN2NEONDataset(
        patch_index=patch_index,
        data_root=data_root,
        split="train",
        training=False,
        return_masks=True,
        max_samples=max_samples,
    )

    print(
        f"Dataset samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Read individual sample
    # --------------------------------------------------------

    sample = dataset[0]

    lr = sample["lr"]
    hr = sample["hr"]

    lr_mask = sample["lr_mask"]
    hr_mask = sample["hr_mask"]

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    assert lr.shape == (
        NUM_BANDS,
        EXPECTED_LR_SIZE,
        EXPECTED_LR_SIZE,
    ), (
        f"Bad LR shape: {lr.shape}"
    )

    assert hr.shape == (
        NUM_BANDS,
        EXPECTED_HR_SIZE,
        EXPECTED_HR_SIZE,
    ), (
        f"Bad HR shape: {hr.shape}"
    )

    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32

    assert lr_mask.dtype == torch.bool
    assert hr_mask.dtype == torch.bool

    # --------------------------------------------------------
    # Range validation
    # --------------------------------------------------------

    if torch.any(
        lr[lr_mask] < 0
    ):
        raise AssertionError(
            "LR contains negative reflectance."
        )

    if torch.any(
        hr[hr_mask] < 0
    ):
        raise AssertionError(
            "HR contains negative reflectance."
        )

    if torch.any(
        lr[lr_mask] > 1.5
    ):
        raise AssertionError(
            "LR contains unexpectedly high "
            "normalized reflectance."
        )

    if torch.any(
        hr[hr_mask] > 1.5
    ):
        raise AssertionError(
            "HR contains unexpectedly high "
            "normalized reflectance."
        )

    # --------------------------------------------------------
    # Scale relationship
    # --------------------------------------------------------

    if (
        hr.shape[-1]
        != lr.shape[-1]
        * EXPECTED_SCALE
    ):
        raise AssertionError(
            "LR/HR scale relationship is incorrect."
        )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    loader = create_dataloader(
        patch_index=patch_index,
        data_root=data_root,
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
        training=False,
        return_masks=True,
        pin_memory=False,
        max_samples=2,
    )

    batch = next(
        iter(loader)
    )

    assert batch["lr"].shape == (
        min(2, len(dataset)),
        NUM_BANDS,
        EXPECTED_LR_SIZE,
        EXPECTED_LR_SIZE,
    )

    assert batch["hr"].shape == (
        min(2, len(dataset)),
        NUM_BANDS,
        EXPECTED_HR_SIZE,
        EXPECTED_HR_SIZE,
    )

    print()
    print(
        f"LR tensor shape : "
        f"{tuple(lr.shape)}"
    )

    print(
        f"HR tensor shape : "
        f"{tuple(hr.shape)}"
    )

    print(
        f"LR dtype        : "
        f"{lr.dtype}"
    )

    print(
        f"HR dtype        : "
        f"{hr.dtype}"
    )

    print(
        f"LR valid pixels : "
        f"{int(lr_mask.sum())}"
    )

    print(
        f"HR valid pixels : "
        f"{int(hr_mask.sum())}"
    )

    print()
    print(
        "PASS: Dataset sample"
    )

    print(
        "PASS: Tensor shapes"
    )

    print(
        "PASS: Tensor dtypes"
    )

    print(
        "PASS: Reflectance range"
    )

    print(
        "PASS: LR/HR 4x relationship"
    )

    print(
        "PASS: DataLoader batching"
    )

    print()
    print(
        "STATUS: DATASET READY"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Test the SEN2NEON PyTorch Dataset."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "SEN2NEON_5GB"
        ),
    )

    parser.add_argument(
        "--patch-index",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    patch_index = (
        args.patch_index
        if args.patch_index is not None
        else (
            args.data_root
            / "patches"
            / "all_patches.csv"
        )
    )

    run_self_test(
        data_root=args.data_root,
        patch_index=patch_index,
        max_samples=args.samples,
    )