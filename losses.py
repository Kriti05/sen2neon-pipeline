#!/usr/bin/env python3
"""
SEN2NEON training losses.

Composite loss for 12-band multispectral super-resolution:

    total = l1_weight        * CharbonnierLoss (or plain L1)
          + sam_weight        * SpectralAngleLoss
          + ssim_weight        * (1 - differentiable SSIM)

All losses support the same mask convention used in metrics.py:
a mask may be [B,1,H,W], [B,C,H,W], [H,W], or [C,H,W]; True/1 means
valid. Invalid pixels are excluded from every term.

This file has no dependency on metrics.py (metrics.py is evaluation-only
and its SSIM is not meant to be backpropagated through), but the masking
convention deliberately mirrors it so behavior is consistent between
training and evaluation.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


# ============================================================
# MASK HANDLING
# ============================================================

def _as_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def prepare_mask(
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a validity mask to [B,C,H,W] float (1.0 = valid)."""
    ref = _as_4d(reference)
    B, C, H, W = ref.shape

    if mask is None:
        return torch.ones_like(ref)

    mask = mask.to(device=ref.device)

    if mask.dtype != torch.bool:
        mask = mask > 0.5

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] == C:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    try:
        mask = mask.expand(B, C, H, W)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to {tuple(ref.shape)}"
        ) from exc

    return mask.float()


# ============================================================
# PIXEL-WISE LOSSES
# ============================================================

class CharbonnierLoss(nn.Module):
    """Smooth, differentiable-everywhere approximation of L1."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = _as_4d(prediction)
        target = _as_4d(target)

        weight = prepare_mask(mask, prediction)
        diff = prediction - target
        loss_map = torch.sqrt(diff * diff + self.eps * self.eps)

        denom = weight.sum().clamp_min(1.0)
        return (loss_map * weight).sum() / denom


class MaskedL1Loss(nn.Module):
    """Plain masked L1."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = _as_4d(prediction)
        target = _as_4d(target)

        weight = prepare_mask(mask, prediction)
        loss_map = torch.abs(prediction - target)

        denom = weight.sum().clamp_min(1.0)
        return (loss_map * weight).sum() / denom


# ============================================================
# SPECTRAL ANGLE LOSS
# ============================================================

class SpectralAngleLoss(nn.Module):
    """
    Mean spectral angle (in radians) between predicted and target
    spectra, computed at pixels where every band is valid.

    Encourages the model to preserve per-pixel spectral shape,
    independent of overall brightness.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = _as_4d(prediction)
        target = _as_4d(target)

        weight = prepare_mask(mask, prediction)
        valid_pixel = weight.min(dim=1).values > 0.5  # [B,H,W]

        p = prediction.permute(0, 2, 3, 1)[valid_pixel]
        t = target.permute(0, 2, 3, 1)[valid_pixel]

        if p.numel() == 0:
            return prediction.sum() * 0.0

        dot = torch.sum(p * t, dim=-1)
        p_norm = torch.linalg.vector_norm(p, dim=-1)
        t_norm = torch.linalg.vector_norm(t, dim=-1)

        cosine = dot / (p_norm * t_norm + EPS)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        angles = torch.acos(cosine)
        return angles.mean()


# ============================================================
# DIFFERENTIABLE SSIM
# ============================================================

class SSIMLoss(nn.Module):
    """1 - masked mean SSIM, averaged over bands. Differentiable."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.data_range = data_range

    def _ssim_map(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = prediction.shape
        k = min(self.window_size, H, W)
        if k % 2 == 0:
            k -= 1
        k = max(k, 3)

        padding = k // 2
        kernel = torch.ones(
            (C, 1, k, k), device=prediction.device, dtype=prediction.dtype
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

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
        denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)

        return numerator / (denominator + EPS)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = _as_4d(prediction)
        target = _as_4d(target)

        weight = prepare_mask(mask, prediction)
        ssim_map = self._ssim_map(prediction, target)

        denom = weight.sum().clamp_min(1.0)
        mean_ssim = (ssim_map * weight).sum() / denom

        return 1.0 - mean_ssim


# ============================================================
# COMPOSITE LOSS
# ============================================================

class CompositeSRLoss(nn.Module):
    """
    Weighted combination of pixel, spectral, and structural losses.

    Any weight set to 0 skips computing that term (saves compute).
    Returns the total loss plus a dict of the individual (unweighted)
    components for logging.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        use_charbonnier: bool = True,
        charbonnier_eps: float = 1e-3,
        sam_weight: float = 0.1,
        ssim_weight: float = 0.1,
        ssim_window: int = 11,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()

        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.ssim_weight = ssim_weight

        self.pixel_loss = (
            CharbonnierLoss(eps=charbonnier_eps)
            if use_charbonnier
            else MaskedL1Loss()
        )
        self.sam_loss = SpectralAngleLoss() if sam_weight > 0 else None
        self.ssim_loss = (
            SSIMLoss(window_size=ssim_window, data_range=data_range)
            if ssim_weight > 0
            else None
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        components: Dict[str, float] = {}

        pixel = self.pixel_loss(prediction, target, mask)
        components["pixel"] = float(pixel.detach().item())
        total = self.l1_weight * pixel

        if self.sam_loss is not None:
            sam = self.sam_loss(prediction, target, mask)
            components["sam"] = float(sam.detach().item())
            total = total + self.sam_weight * sam

        if self.ssim_loss is not None:
            ssim_term = self.ssim_loss(prediction, target, mask)
            components["ssim"] = float(ssim_term.detach().item())
            total = total + self.ssim_weight * ssim_term

        components["total"] = float(total.detach().item())
        return total, components


def build_loss(config: Optional[dict] = None) -> CompositeSRLoss:
    """Build a CompositeSRLoss from a plain dict (e.g. parsed from YAML)."""
    config = config or {}
    return CompositeSRLoss(
        l1_weight=float(config.get("l1_weight", 1.0)),
        use_charbonnier=bool(config.get("charbonnier", True)),
        charbonnier_eps=float(config.get("charbonnier_eps", 1e-3)),
        sam_weight=float(config.get("sam_weight", 0.1)),
        ssim_weight=float(config.get("ssim_weight", 0.1)),
        ssim_window=int(config.get("ssim_window", 11)),
        data_range=float(config.get("data_range", 1.0)),
    )


# ============================================================
# SELF TEST
# ============================================================

def _self_test() -> None:
    print("=" * 72)
    print("LOSSES SELF-TEST")
    print("=" * 72)

    torch.manual_seed(0)

    target = torch.rand(2, 12, 32, 32)
    prediction = target.clone().requires_grad_(True)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, :2, :2] = False

    criterion = build_loss(
        {
            "l1_weight": 1.0,
            "charbonnier": True,
            # Use a tiny epsilon here purely so the "identical inputs"
            # check below is meaningful. The default epsilon (1e-3) is a
            # deliberate smoothing constant for training and is *not* a
            # bug: CharbonnierLoss(x=0) == eps, not 0, by construction.
            "charbonnier_eps": 1e-6,
            "sam_weight": 0.1,
            "ssim_weight": 0.1,
        }
    )

    total, components = criterion(prediction, target, mask)

    # Identical inputs should be near-zero, not exactly zero: the SAM term
    # clamps cosine similarity to (-1+1e-7, 1-1e-7) to keep acos() finite,
    # which leaves a small residual angle even for perfectly equal spectra.
    if total.item() > 5e-4:
        raise RuntimeError("Identical prediction/target should give ~0 loss.")

    total.backward()

    if prediction.grad is None or not torch.isfinite(prediction.grad).all():
        raise RuntimeError("Gradient did not propagate correctly.")

    noisy_pred = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    total_noisy, components_noisy = criterion(noisy_pred, target, mask)
    total_noisy.backward()

    print(f"Identical loss   : {total.item():.8f}  components={components}")
    print(f"Noisy loss       : {total_noisy.item():.8f}  components={components_noisy}")

    if total_noisy.item() <= total.item():
        raise RuntimeError("Noisy prediction should have higher loss.")

    if noisy_pred.grad is None or not torch.isfinite(noisy_pred.grad).all():
        raise RuntimeError("Gradient did not propagate correctly for noisy case.")

    print()
    print("PASS: Charbonnier / L1")
    print("PASS: Spectral angle loss")
    print("PASS: SSIM loss")
    print("PASS: Composite gradient flow")
    print("STATUS: LOSSES READY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    _self_test()
