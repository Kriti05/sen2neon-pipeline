#!/usr/bin/env python3
"""
SEN2NEON Baseline CNN
=====================

12-band 4x multispectral super-resolution model.

Input:
    [B, 12, 64, 64]

Output:
    [B, 12, 256, 256]

Architecture:
    Input
      |
      v
    Conv
      |
      v
    Residual CNN blocks
      |
      v
    2x PixelShuffle
      |
      v
    2x PixelShuffle
      |
      v
    Residual reconstruction
      |
      +------ Bicubic LR
      |
      v
    HR output

The model predicts a residual over bicubic interpolation.

This is intended to be the first reproducible baseline before
testing Transformer and hybrid architectures.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_IN_CHANNELS = 12
DEFAULT_OUT_CHANNELS = 12
DEFAULT_FEATURES = 64
DEFAULT_BLOCKS = 8
DEFAULT_SCALE = 4


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    """
    Standard residual convolution block.

    Structure:
        Conv -> ReLU -> Conv -> residual addition
    """

    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.1,
    ) -> None:

        super().__init__()

        self.residual_scale = (
            residual_scale
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.activation = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.conv1(x)

        residual = self.activation(
            residual
        )

        residual = self.conv2(
            residual
        )

        return (
            x
            + self.residual_scale
            * residual
        )


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class PixelShuffleBlock(nn.Module):
    """
    2x learned upsampling.

    Conv:
        C -> 4C

    PixelShuffle(2):
        spatial resolution x2
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pixel_shuffle = (
            nn.PixelShuffle(2)
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.pixel_shuffle(
            x
        )

        x = self.activation(
            x
        )

        return x


# ============================================================
# BASELINE CNN
# ============================================================

class BaselineCNN(nn.Module):
    """
    Multispectral residual CNN baseline.

    Parameters
    ----------
    in_channels:
        Number of input spectral bands.

    out_channels:
        Number of output spectral bands.

    features:
        Internal feature width.

    num_blocks:
        Number of residual CNN blocks.

    scale:
        Super-resolution scale.

    residual_scale:
        Scaling applied inside residual blocks.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        features: int = DEFAULT_FEATURES,
        num_blocks: int = DEFAULT_BLOCKS,
        scale: int = DEFAULT_SCALE,
        residual_scale: float = 0.1,
    ) -> None:

        super().__init__()

        if scale != 4:
            raise ValueError(
                "This baseline is configured "
                "for 4x super-resolution."
            )

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be > 0."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be > 0."
            )

        if features <= 0:
            raise ValueError(
                "features must be > 0."
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be > 0."
            )

        # ----------------------------------------------------
        # Shallow feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ----------------------------------------------------
        # Deep residual feature extraction
        # ----------------------------------------------------

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    channels=features,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ----------------------------------------------------
        # 4x upsampling = 2x + 2x
        # ----------------------------------------------------

        self.upsample = nn.Sequential(
            PixelShuffleBlock(
                features
            ),
            PixelShuffleBlock(
                features
            ),
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(
        self,
    ) -> None:

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape "
                "[B,C,H,W]. "
                f"Got {tuple(x.shape)}"
            )

        if x.shape[1] != self.head.in_channels:
            raise ValueError(
                f"Expected "
                f"{self.head.in_channels} "
                f"input channels, got "
                f"{x.shape[1]}."
            )

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = self.head(x)

        body = self.body(
            features
        )

        body = self.body_conv(
            body
        )

        # Global feature residual
        features = (
            features + body
        )

        # ----------------------------------------------------
        # Learned 4x upsampling
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        # ----------------------------------------------------
        # Predict HR residual
        # ----------------------------------------------------

        residual = self.tail(
            features
        )

        # ----------------------------------------------------
        # Bicubic + learned residual
        # ----------------------------------------------------

        output = (
            bicubic + residual
        )

        return output


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model: nn.Module,
) -> Tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# MODEL FACTORY
# ============================================================

def create_baseline_model(
    in_channels: int = 12,
    out_channels: int = 12,
    features: int = 64,
    num_blocks: int = 8,
    scale: int = 4,
) -> BaselineCNN:

    return BaselineCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        num_blocks=num_blocks,
        scale=scale,
    )


# ============================================================
# SELF TEST
# ============================================================

def self_test(
    device: str = "cpu",
) -> None:

    print()
    print("=" * 72)
    print("BASELINE CNN SELF-TEST")
    print("=" * 72)

    model = create_baseline_model()

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    x = torch.rand(
        2,
        12,
        64,
        64,
        device=device,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_shape = (
        2,
        12,
        256,
        256,
    )

    if tuple(y.shape) != expected_shape:

        raise RuntimeError(
            f"Wrong output shape: "
            f"{tuple(y.shape)}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # NaN / Inf check
    # --------------------------------------------------------

    if not torch.isfinite(y).all():

        raise RuntimeError(
            "Model output contains "
            "NaN or Inf values."
        )

    # --------------------------------------------------------
    # Gradient test
    # --------------------------------------------------------

    model.train()

    x_train = torch.rand(
        1,
        12,
        64,
        64,
        device=device,
        requires_grad=True,
    )

    y_train = model(
        x_train
    )

    target = torch.rand_like(
        y_train
    )

    loss = F.l1_loss(
        y_train,
        target,
    )

    loss.backward()

    if x_train.grad is None:

        raise RuntimeError(
            "Gradient did not propagate "
            "to model input."
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total, trainable = (
        count_parameters(model)
    )

    print(
        f"Input shape     : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape    : "
        f"{tuple(y.shape)}"
    )

    print(
        f"Parameters      : "
        f"{total:,}"
    )

    print(
        f"Trainable       : "
        f"{trainable:,}"
    )

    print(
        f"Test loss       : "
        f"{loss.item():.6f}"
    )

    print()
    print(
        "PASS: Forward pass"
    )

    print(
        "PASS: 4x output resolution"
    )

    print(
        "PASS: 12-band preservation"
    )

    print(
        "PASS: Finite output"
    )

    print(
        "PASS: Backpropagation"
    )

    print()
    print(
        "STATUS: BASELINE CNN READY"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Test the SEN2NEON baseline CNN."
        )
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.device == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA requested but "
            "CUDA is not available."
        )

    print(
        f"Using device: {device}"
    )

    self_test(
        device=device
    )


if __name__ == "__main__":
    main()