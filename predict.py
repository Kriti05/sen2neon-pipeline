"""
predict.py — Inference module for SEN2NEON 4x super-resolution.

Loads the trained HybridCNNTransformerSR checkpoint, applies the same
normalization used during training, runs the model (with tiling for large
inputs), and returns the enhanced 4x image plus a post-hoc uncertainty map.

The model requires H and W to be divisible by window_size (8). Large tiles
are split into overlapping 64x64 patches and stitched with a Hann window
to avoid seams and OOM.

Usage:
    from predict import build_model_from_checkpoint, load_norm_stats, predict_tile
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.baseline_cnn import BaselineCNN
from models.transformer import TransformerSR
from models.proposed_model import HybridCNNTransformerSR


WINDOW_SIZE = 8      # must match HybridCNNTransformerSR.window_size
SCALE = 4
PATCH_SIZE = 64      # spatial size the model expects at inference
OVERLAP = 8          # overlap between tiled patches


# ============================================================
# MODEL LOADING
# ============================================================

def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, str, dict]:
    """Rebuild the model described by the checkpoint and load its weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "hybrid")
    cfg = ckpt.get("model_config", {})

    if model_name == "cnn":
        model = BaselineCNN(**cfg)
    elif model_name == "transformer":
        model = TransformerSR(**cfg)
    elif model_name == "hybrid":
        model = HybridCNNTransformerSR(**cfg)
    else:
        raise ValueError(f"Unknown model name in checkpoint: {model_name}")

    state = ckpt.get("model_state_dict", ckpt)
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys while loading checkpoint.")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys while loading checkpoint.")

    model.to(device).eval()
    return model, model_name, cfg


# ============================================================
# NORMALIZATION
# ============================================================

def load_norm_stats(data_root: Path) -> Optional[dict]:
    """
    Search common locations for the normalization stats written by
    preprocessing/normalize.py. Returns None if nothing is found.
    """
    candidates = [
        data_root / "patches" / "norm_stats.json",
        data_root / "patches" / "normalization.json",
        data_root / "normalization" / "stats.json",
        data_root / "normalization" / "norm_stats.json",
        data_root / "norm_stats.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                stats = json.loads(path.read_text())
                # Accept either {"mean": [...], "std": [...]} or {"bands": {...}}
                if "mean" in stats and "std" in stats:
                    return stats
                if "bands" in stats:
                    means = [stats["bands"][b]["mean"] for b in stats["bands"]]
                    stds = [stats["bands"][b]["std"] for b in stats["bands"]]
                    return {"mean": means, "std": stds}
            except Exception as exc:
                print(f"WARNING: could not parse {path}: {exc}")
    return None


def normalize(img: np.ndarray, stats: Optional[dict]) -> np.ndarray:
    """[C,H,W] float32 → normalized [C,H,W]."""
    if stats is None:
        return (img / 10000.0).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    return ((img - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def denormalize(img: np.ndarray, stats: Optional[dict]) -> np.ndarray:
    """[C,H,W] normalized → [C,H,W] raw reflectance scale."""
    if stats is None:
        return (img * 10000.0).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    return (img * std + mean).astype(np.float32)


# ============================================================
# PADDING
# ============================================================

def pad_to_window_multiple(
    x: torch.Tensor,
    window_size: int = WINDOW_SIZE,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad H,W up to the nearest multiple of window_size (model requirement).
    Returns (padded, (pad_h, pad_w)).
    """
    _, _, h, w = x.shape
    H = ((h + window_size - 1) // window_size) * window_size
    W = ((w + window_size - 1) // window_size) * window_size
    pad_h, pad_w = H - h, W - w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


# ============================================================
# UNCERTAINTY (post-hoc, gradient energy)
# ============================================================

def estimate_uncertainty(enhanced_norm: np.ndarray) -> np.ndarray:
    """
    Post-hoc uncertainty: local gradient energy of the enhanced image.
    High-gradient regions are where the model invented sharp detail.
    Returns [H*4, W*4] in [0,1].
    """
    gray = enhanced_norm.mean(axis=0).astype(np.float32)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    mx = float(grad.max())
    if mx > 1e-8:
        uncertainty = grad / mx
    else:
        uncertainty = np.zeros_like(grad)
    # Optional smoothing to make the map visually meaningful
    try:
        from scipy.ndimage import gaussian_filter
        uncertainty = gaussian_filter(uncertainty, sigma=2.0)
    except ImportError:
        pass
    return uncertainty.astype(np.float32)


# ============================================================
# SINGLE-PATCH INFERENCE
# ============================================================

@torch.no_grad()
def _forward_patch(
    model: torch.nn.Module,
    patch_norm: np.ndarray,      # [C, h, w] normalized, h,w <= 64
    device: torch.device,
) -> np.ndarray:
    """Run the model on a single normalized patch, return normalized output [C,h*4,w*4]."""
    x = torch.from_numpy(patch_norm).float().unsqueeze(0).to(device)
    x_padded, (pad_h, pad_w) = pad_to_window_multiple(x, WINDOW_SIZE)
    pred = model(x_padded)
    if pad_h or pad_w:
        h4 = x.shape[-2] * SCALE
        w4 = x.shape[-1] * SCALE
        pred = pred[..., :h4, :w4]
    return pred.squeeze(0).cpu().numpy()


# ============================================================
# PUBLIC ENTRY POINT — handles any tile size
# ============================================================

@torch.no_grad()
def predict_tile(
    model: torch.nn.Module,
    tile: np.ndarray,                    # [12, H, W] raw reflectance
    stats: Optional[dict],
    device: torch.device,
    already_normalized: bool = False,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    progress_callback=None,              # optional callable(done, total)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance a full tile of any size.

    Small tiles (<= patch_size in both dims) run in a single forward pass.
    Larger tiles are processed patch-by-patch with Hann-window blending to
    avoid seams.

    Returns:
        enhanced:    [12, H*4, W*4] float32, raw reflectance scale
        uncertainty: [H*4, W*4] float32 in [0,1]
    """
    if tile.ndim != 3:
        raise ValueError(f"Expected tile shape [C,H,W], got {tile.shape}")
    if tile.shape[0] != 12 and tile.shape[-1] == 12:
        # Accept [H,W,C] too
        tile = tile.transpose(2, 0, 1)

    tile_n = tile if already_normalized else normalize(tile, stats)
    C, H, W = tile_n.shape

    # ------- Small tile: single pass -------
    if H <= patch_size and W <= patch_size:
        out_n = _forward_patch(model, tile_n, device)
        enhanced = denormalize(out_n, stats)
        return enhanced, estimate_uncertainty(out_n)

    # ------- Large tile: tiled inference with Hann blending -------
    stride = max(patch_size - overlap, 1)
    out_n = np.zeros((C, H * SCALE, W * SCALE), dtype=np.float32)
    weight = np.zeros((H * SCALE, W * SCALE), dtype=np.float32)

    # Precompute Hann window in HR space
    def hann_weight(ph: int, pw: int) -> np.ndarray:
        wy = np.hanning(ph * SCALE).astype(np.float32)
        wx = np.hanning(pw * SCALE).astype(np.float32)
        return np.outer(wy, wx)

    y_starts = list(range(0, max(H - patch_size, 0) + stride, stride))
    if not y_starts or y_starts[-1] + patch_size < H:
        y_starts.append(max(H - patch_size, 0))
    x_starts = list(range(0, max(W - patch_size, 0) + stride, stride))
    if not x_starts or x_starts[-1] + patch_size < W:
        x_starts.append(max(W - patch_size, 0))

    total = len(y_starts) * len(x_starts)
    done = 0

    for y in y_starts:
        for x in x_starts:
            y2 = min(y + patch_size, H)
            x2 = min(x + patch_size, W)
            y1 = max(y2 - patch_size, 0)
            x1 = max(x2 - patch_size, 0)

            patch = tile_n[:, y1:y2, x1:x2]      # [C, ph, pw]
            ph, pw = patch.shape[-2:]

            pred_n = _forward_patch(model, patch, device)   # [C, ph*4, pw*4]

            w = hann_weight(ph, pw)
            oy1, oy2 = y1 * SCALE, y2 * SCALE
            ox1, ox2 = x1 * SCALE, x2 * SCALE
            out_n[:, oy1:oy2, ox1:ox2] += pred_n * w
            weight[oy1:oy2, ox1:ox2] += w

            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    weight = np.maximum(weight, 1e-6)
    out_n = out_n / weight[None, :, :]
    enhanced = denormalize(out_n, stats)
    return enhanced, estimate_uncertainty(out_n)


# ============================================================
# GEOTIFF I/O
# ============================================================

def load_geotiff(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a 12-band GeoTIFF; return [C,H,W] float32 and metadata."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }
    return arr, meta


def save_geotiff(
    path: Path,
    arr: np.ndarray,
    ref_meta: dict,
    scale: int = SCALE,
) -> None:
    """Write an enhanced array as a GeoTIFF, scaling the affine transform."""
    import rasterio
    from rasterio.transform import Affine

    t = ref_meta["transform"]
    new_transform = Affine(t[0] / scale, t[1], t[2],
                           t[3], t[4] / scale, t[5])

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=ref_meta.get("crs"),
        transform=new_transform,
        compress="lzw",
    ) as dst:
        dst.write(arr)


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def _stretch(band: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(band, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.clip(band, 0, 1)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def to_rgb(
    arr: np.ndarray,
    bands: Tuple[int, int, int] = (3, 2, 1),   # B4, B3, B2 (1-indexed → 0-indexed)
    stretch: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """[C,H,W] reflectance → [H,W,3] uint8 true-color preview."""
    rgb = np.stack([arr[b] for b in bands], axis=0)
    out = np.stack([_stretch(rgb[i], *stretch) for i in range(3)], axis=-1)
    return (out * 255).astype(np.uint8)


def to_false_color(
    arr: np.ndarray,
    bands: Tuple[int, int, int] = (7, 3, 2),   # B8, B4, B3
    stretch: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """NIR-red-green false color (vegetation appears red)."""
    return to_rgb(arr, bands=bands, stretch=stretch)


def compute_ndvi(arr: np.ndarray) -> np.ndarray:
    """NDVI from B8 (NIR) and B4 (red); returns [H,W] in [-1,1]."""
    nir = arr[7]     # B8
    red = arr[3]     # B4
    denom = nir + red
    ndvi = np.where(np.abs(denom) > 1e-6, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def ndvi_to_rgb(ndvi: np.ndarray) -> np.ndarray:
    """Colored NDVI preview (red → low, green → high)."""
    norm = np.clip((ndvi + 1) / 2, 0, 1)          # [-1,1] → [0,1]
    # Simple red-yellow-green ramp
    r = np.clip(2 * (1 - norm), 0, 1)
    g = np.clip(2 * norm, 0, 1)
    b = np.zeros_like(norm)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)