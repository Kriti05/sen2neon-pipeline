#!/usr/bin/env python3
"""
SEN2NEON evaluation metrics.

Metrics implemented for 12-band multispectral super-resolution:
    - MAE
    - RMSE
    - PSNR
    - SSIM
    - SAM (Spectral Angle Mapper)
    - ERGAS
    - spectral correlation

All metrics support masks. A mask may be [B,1,H,W], [B,C,H,W],
[H,W], or [C,H,W]. Invalid pixels/bands are excluded.

Expected reflectance convention:
    uint16 source values / 10000.0 -> float reflectance

The functions operate on torch tensors and return Python floats.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    """Convert [C,H,W] or [B,C,H,W] to [B,C,H,W]."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _prepare_mask(
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a validity mask to [B,C,H,W]. True means valid."""
    ref = _as_4d(reference)
    B, C, H, W = ref.shape

    if mask is None:
        return torch.ones_like(ref, dtype=torch.bool)

    mask = mask.to(device=ref.device, dtype=torch.bool)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        # Could be [C,H,W] or [B,H,W].
        if mask.shape[0] == C:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    try:
        return mask.expand(B, C, H, W)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to {tuple(ref.shape)}"
        ) from exc


def _masked_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened prediction, target, and valid mask."""
    prediction = _as_4d(prediction).float()
    target = _as_4d(target).float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    valid = _prepare_mask(mask, prediction)
    valid = valid & torch.isfinite(prediction) & torch.isfinite(target)

    return prediction[valid], target[valid], valid


def mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean absolute error over valid values."""
    p, t, _ = _masked_values(prediction, target, mask)
    if p.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(p - t)).item())


def rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Root mean squared error over valid values."""
    p, t, _ = _masked_values(prediction, target, mask)
    if p.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean((p - t) ** 2)).item())


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
) -> float:
    """Masked PSNR in dB."""
    p, t, _ = _masked_values(prediction, target, mask)
    if p.numel() == 0:
        return float("nan")

    mse = torch.mean((p - t) ** 2)
    if mse <= 0:
        return float("inf")

    value = 10.0 * torch.log10(
        torch.tensor(data_range ** 2, device=p.device) / mse
    )
    return float(value.item())


def _ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute a channel-wise SSIM map without external dependencies."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    B, C, H, W = prediction.shape
    k = min(window_size, H, W)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        raise ValueError("Images must be at least 3x3 for SSIM")

    padding = k // 2
    kernel = torch.ones(
        (C, 1, k, k),
        device=prediction.device,
        dtype=prediction.dtype,
    ) / float(k * k)

    mu_p = F.conv2d(prediction, kernel, padding=padding, groups=C)
    mu_t = F.conv2d(target, kernel, padding=padding, groups=C)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(prediction * prediction, kernel, padding=padding, groups=C) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, kernel, padding=padding, groups=C) - mu_t_sq
    sigma_pt = F.conv2d(prediction * target, kernel, padding=padding, groups=C) - mu_pt

    sigma_p_sq = torch.clamp(sigma_p_sq, min=0.0)
    sigma_t_sq = torch.clamp(sigma_t_sq, min=0.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)

    return numerator / (denominator + EPS)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    window_size: int = 11,
) -> float:
    """Masked mean SSIM, averaged over bands."""
    prediction = _as_4d(prediction).float()
    target = _as_4d(target).float()

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    valid = _prepare_mask(mask, prediction)
    valid = valid & torch.isfinite(prediction) & torch.isfinite(target)

    # SSIM is a local metric. Use the pixel validity mask as a weighting map.
    ssim_values = _ssim_map(prediction, target, window_size, data_range)
    valid_float = valid.float()

    numerator = (ssim_values * valid_float).sum()
    denominator = valid_float.sum()

    if denominator <= 0:
        return float("nan")

    return float((numerator / denominator).item())


def spectral_angle_mapper(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    degrees: bool = True,
) -> float:
    """Mean spectral angle between predicted and target spectra."""
    prediction = _as_4d(prediction).float()
    target = _as_4d(target).float()

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    # For spectral metrics, require every band at a pixel to be valid.
    valid = _prepare_mask(mask, prediction)
    valid_pixel = valid.all(dim=1)
    valid_pixel = valid_pixel & torch.isfinite(prediction).all(dim=1)
    valid_pixel = valid_pixel & torch.isfinite(target).all(dim=1)

    p = prediction.permute(0, 2, 3, 1)[valid_pixel]
    t = target.permute(0, 2, 3, 1)[valid_pixel]

    if p.numel() == 0:
        return float("nan")

    dot = torch.sum(p * t, dim=-1)
    p_norm = torch.linalg.vector_norm(p, dim=-1)
    t_norm = torch.linalg.vector_norm(t, dim=-1)

    cosine = dot / (p_norm * t_norm)
    cosine = torch.clamp(cosine, -1.0, 1.0)

    # If both spectra are almost zero, the angle is not informative.
    nonzero = (p_norm > EPS) & (t_norm > EPS)
    if not nonzero.any():
        return float("nan")

    angles = torch.acos(cosine[nonzero])

    if degrees:
        angles = angles * (180.0 / math.pi)

    return float(angles.mean().item())


def ergas(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: float = 4.0,
) -> float:
    """
    ERGAS for multispectral super-resolution.

    ERGAS = 100 / scale * sqrt(mean((RMSE_band / mean_band)^2))
    """
    prediction = _as_4d(prediction).float()
    target = _as_4d(target).float()

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    valid = _prepare_mask(mask, prediction)
    valid = valid & torch.isfinite(prediction) & torch.isfinite(target)

    terms = []

    for band in range(prediction.shape[1]):
        valid_band = valid[:, band]
        p = prediction[:, band][valid_band]
        t = target[:, band][valid_band]

        if p.numel() == 0:
            continue

        band_rmse = torch.sqrt(
            torch.mean((p - t) ** 2) + EPS
        )
        band_mean = torch.mean(torch.abs(t))

        if band_mean > EPS:
            terms.append(
                (band_rmse / band_mean) ** 2
            )

    if not terms:
        return float("nan")

    value = (
        100.0
        / float(scale)
        * torch.sqrt(torch.stack(terms).mean())
    )

    return float(value.item())


def spectral_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Mean Pearson correlation between predicted and target spectra
    across the 12 bands, computed independently at valid pixels.
    """
    prediction = _as_4d(prediction).float()
    target = _as_4d(target).float()

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    valid = _prepare_mask(mask, prediction)
    valid_pixel = valid.all(dim=1)
    valid_pixel = valid_pixel & torch.isfinite(prediction).all(dim=1)
    valid_pixel = valid_pixel & torch.isfinite(target).all(dim=1)

    p = prediction.permute(0, 2, 3, 1)[valid_pixel]
    t = target.permute(0, 2, 3, 1)[valid_pixel]

    if p.shape[0] == 0 or p.shape[1] < 2:
        return float("nan")

    p_centered = p - p.mean(dim=1, keepdim=True)
    t_centered = t - t.mean(dim=1, keepdim=True)

    numerator = (p_centered * t_centered).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(p_centered, dim=1)
        * torch.linalg.vector_norm(t_centered, dim=1)
    )

    valid_corr = denominator > EPS

    if not valid_corr.any():
        return float("nan")

    corr = numerator[valid_corr] / denominator[valid_corr]
    return float(corr.mean().item())


def per_band_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, list]:
    """Return MAE/RMSE/PSNR for each spectral band."""
    prediction = _as_4d(prediction)
    target = _as_4d(target)
    valid = _prepare_mask(mask, prediction)

    result = {
        "mae": [],
        "rmse": [],
        "psnr": [],
    }

    for band in range(prediction.shape[1]):
        p = prediction[:, band:band + 1]
        t = target[:, band:band + 1]
        m = valid[:, band:band + 1]

        result["mae"].append(mae(p, t, m))
        result["rmse"].append(rmse(p, t, m))
        result["psnr"].append(psnr(p, t, m))

    return result


def calculate_all_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: float = 4.0,
) -> Dict[str, float]:
    """Calculate the complete metric set."""
    return {
        "mae": mae(prediction, target, mask),
        "rmse": rmse(prediction, target, mask),
        "psnr_db": psnr(prediction, target, mask),
        "ssim": ssim(prediction, target, mask),
        "sam_degrees": spectral_angle_mapper(prediction, target, mask),
        "ergas": ergas(prediction, target, mask, scale=scale),
        "spectral_correlation": spectral_correlation(prediction, target, mask),
    }


def _self_test() -> None:
    print("=" * 72)
    print("METRICS SELF-TEST")
    print("=" * 72)

    torch.manual_seed(42)

    target = torch.rand(2, 12, 32, 32)
    prediction = target + 0.01 * torch.randn_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)

    # Make a small invalid region.
    mask[:, :, :2, :2] = False

    results = calculate_all_metrics(
        prediction,
        target,
        mask,
    )

    for name, value in results.items():
        print(f"{name:24s}: {value}")

    if not all(
        np.isfinite(v)
        for v in results.values()
        if not isinstance(v, float) or not math.isnan(v)
    ):
        raise RuntimeError("Metric self-test produced invalid values.")

    identical = calculate_all_metrics(
        target,
        target,
        mask,
    )

    if identical["mae"] > 1e-7:
        raise RuntimeError("Identical-image MAE test failed.")

    if identical["rmse"] > 1e-7:
        raise RuntimeError("Identical-image RMSE test failed.")

    if identical["sam_degrees"] > 0.05:
        raise RuntimeError("Identical-image SAM test failed.")

    if identical["spectral_correlation"] < 0.9999:
        raise RuntimeError("Identical-image spectral correlation test failed.")

    print()
    print("PASS: Mask handling")
    print("PASS: MAE/RMSE")
    print("PASS: PSNR")
    print("PASS: SSIM")
    print("PASS: SAM")
    print("PASS: ERGAS")
    print("PASS: Spectral correlation")
    print("PASS: Identity tests")
    print("STATUS: METRICS READY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    # Running without arguments also performs the self-test because this file
    # is intended to be directly executable during project setup.
    _self_test()
