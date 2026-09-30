#!/usr/bin/env python3
"""
SEN2NEON Transformer Baseline
=============================

Lightweight window-attention Transformer for
12-band 4x multispectral super-resolution.

Input:
    [B, 12, 64, 64]

Output:
    [B, 12, 256, 256]

Design:
    12-band input
        |
        v
    Shallow CNN embedding
        |
        v
    Window Transformer blocks
        |
        v
    Feature reconstruction
        |
        v
    4x PixelShuffle
        |
        v
    12-band HR output

Attention is performed inside local windows rather than
globally, avoiding O((H*W)^2) attention.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_IN_CHANNELS = 12
DEFAULT_OUT_CHANNELS = 12
DEFAULT_EMBED_DIM = 96
DEFAULT_DEPTH = 4
DEFAULT_NUM_HEADS = 4
DEFAULT_WINDOW_SIZE = 8
DEFAULT_SCALE = 4


# ============================================================
# WINDOW PARTITION
# ============================================================

def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Convert feature map into local attention windows.

    Input:
        [B, H, W, C]

    Output:
        [B*num_windows, window_size*window_size, C]
    """

    if x.ndim != 4:
        raise ValueError(
            "window_partition expects "
            "[B,H,W,C]."
        )

    B, H, W, C = x.shape

    if H % window_size != 0:
        raise ValueError(
            f"Height {H} is not divisible "
            f"by window size {window_size}."
        )

    if W % window_size != 0:
        raise ValueError(
            f"Width {W} is not divisible "
            f"by window size {window_size}."
        )

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    windows = x.view(
        -1,
        window_size * window_size,
        C,
    )

    return windows


# ============================================================
# WINDOW REVERSE
# ============================================================

def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Reverse window partition.

    Input:
        [B*num_windows, tokens, C]

    Output:
        [B,H,W,C]
    """

    if windows.ndim != 3:
        raise ValueError(
            "window_reverse expects "
            "[num_windows,tokens,C]."
        )

    C = windows.shape[-1]

    windows_per_image = (
        height // window_size
    ) * (
        width // window_size
    )

    expected_windows = (
        batch_size
        * windows_per_image
    )

    if windows.shape[0] != expected_windows:
        raise ValueError(
            "Number of windows does not "
            "match the supplied dimensions."
        )

    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        C,
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    x = x.view(
        batch_size,
        height,
        width,
        C,
    )

    return x


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):
    """
    Transformer feed-forward network.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        hidden_dim = int(
            dim * expansion
        )

        self.fc1 = nn.Linear(
            dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            dim,
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        x = self.dropout2(x)

        return x


# ============================================================
# WINDOW TRANSFORMER BLOCK
# ============================================================

class WindowTransformerBlock(
    nn.Module
):
    """
    Local window Transformer block.

    Structure:

        LayerNorm
            |
            v
        Window MHA
            |
            v
        Residual
            |
        LayerNorm
            |
            v
           MLP
            |
            v
        Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "embed dimension must be "
                "divisible by num_heads."
            )

        self.dim = dim

        self.window_size = (
            window_size
        )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.mlp = MLP(
            dim=dim,
            expansion=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Transformer block expects "
                "[B,H,W,C]."
            )

        B, H, W, C = x.shape

        if C != self.dim:
            raise ValueError(
                f"Expected {self.dim} channels, "
                f"got {C}."
            )

        # ----------------------------------------------------
        # Local attention
        # ----------------------------------------------------

        residual = x

        x_norm = self.norm1(x)

        windows = window_partition(
            x_norm,
            self.window_size,
        )

        attended, _ = self.attention(
            windows,
            windows,
            windows,
            need_weights=False,
        )

        attended = window_reverse(
            attended,
            window_size=self.window_size,
            height=H,
            width=W,
            batch_size=B,
        )

        x = residual + attended

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        x = (
            x
            + self.mlp(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class TransformerUpsampleBlock(
    nn.Module
):
    """
    Learned 2x upsampling.
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
            padding=1,
        )

        self.shuffle = nn.PixelShuffle(
            2
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.shuffle(x)

        x = self.activation(x)

        return x


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class TransformerSR(nn.Module):
    """
    Window-attention Transformer
    for 12-band 4x super-resolution.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        embed_dim: int = DEFAULT_EMBED_DIM,
        depth: int = DEFAULT_DEPTH,
        num_heads: int = DEFAULT_NUM_HEADS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        scale: int = DEFAULT_SCALE,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if scale != 4:
            raise ValueError(
                "TransformerSR currently "
                "supports only 4x scaling."
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible "
                "by num_heads."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be > 0."
            )

        self.in_channels = (
            in_channels
        )

        self.out_channels = (
            out_channels
        )

        self.embed_dim = embed_dim

        self.window_size = (
            window_size
        )

        # ----------------------------------------------------
        # Input embedding
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                WindowTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.body_conv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsample = nn.Sequential(
            TransformerUpsampleBlock(
                embed_dim
            ),
            TransformerUpsampleBlock(
                embed_dim
            ),
        )

        # ----------------------------------------------------
        # Output reconstruction
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            embed_dim,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Initialization
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

            elif isinstance(
                module,
                nn.Linear,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
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
                "[B,C,H,W]."
            )

        B, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected "
                f"{self.in_channels} channels, "
                f"got {C}."
            )

        if (
            H % self.window_size != 0
            or W % self.window_size != 0
        ):

            raise ValueError(
                f"Input spatial dimensions "
                f"must be divisible by "
                f"window_size={self.window_size}. "
                f"Got {H}x{W}."
            )

        # ----------------------------------------------------
        # CNN embedding
        # ----------------------------------------------------

        features = self.head(x)

        # ----------------------------------------------------
        # Convert:
        #
        # [B,C,H,W]
        #
        # to:
        #
        # [B,H,W,C]
        # ----------------------------------------------------

        features = (
            features
            .permute(
                0,
                2,
                3,
                1,
            )
            .contiguous()
        )

        transformer_input = (
            features
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:

            features = block(
                features
            )

        features = self.norm(
            features
        )

        # ----------------------------------------------------
        # Return to CNN layout
        # ----------------------------------------------------

        features = (
            features
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        features = self.body_conv(
            features
        )

        # Global residual connection
        shallow = (
            transformer_input
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        features = (
            features + shallow
        )

        # ----------------------------------------------------
        # 4x learned upsampling
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        # ----------------------------------------------------
        # HR reconstruction
        # ----------------------------------------------------

        output = self.tail(
            features
        )

        return output


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model: nn.Module,
) -> Tuple[int, int]:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# MODEL FACTORY
# ============================================================

def create_transformer_model(
    in_channels: int = 12,
    out_channels: int = 12,
    embed_dim: int = 96,
    depth: int = 4,
    num_heads: int = 4,
    window_size: int = 8,
    scale: int = 4,
) -> TransformerSR:

    return TransformerSR(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
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
    print("TRANSFORMER SR SELF-TEST")
    print("=" * 72)

    model = create_transformer_model()

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
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    # --------------------------------------------------------
    # Shape
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
    # Numerical stability
    # --------------------------------------------------------

    if not torch.isfinite(y).all():

        raise RuntimeError(
            "Output contains NaN or Inf."
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
            "Gradient did not propagate."
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total, trainable = (
        count_parameters(model)
    )

    # --------------------------------------------------------
    # Attention complexity information
    # --------------------------------------------------------

    tokens_per_window = (
        DEFAULT_WINDOW_SIZE
        * DEFAULT_WINDOW_SIZE
    )

    print(
        f"Input shape          : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape         : "
        f"{tuple(y.shape)}"
    )

    print(
        f"Embedding dimension  : "
        f"{DEFAULT_EMBED_DIM}"
    )

    print(
        f"Transformer blocks   : "
        f"{DEFAULT_DEPTH}"
    )

    print(
        f"Attention heads      : "
        f"{DEFAULT_NUM_HEADS}"
    )

    print(
        f"Window size          : "
        f"{DEFAULT_WINDOW_SIZE}x"
        f"{DEFAULT_WINDOW_SIZE}"
    )

    print(
        f"Tokens/window        : "
        f"{tokens_per_window}"
    )

    print(
        f"Parameters           : "
        f"{total:,}"
    )

    print(
        f"Trainable            : "
        f"{trainable:,}"
    )

    print(
        f"Test loss            : "
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
        "PASS: 12-band output"
    )

    print(
        "PASS: Window attention"
    )

    print(
        "PASS: Finite output"
    )

    print(
        "PASS: Backpropagation"
    )

    print()
    print(
        "STATUS: TRANSFORMER READY"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Test the SEN2NEON "
            "Transformer SR model."
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
            "CUDA is unavailable."
        )

    print(
        f"Using device: {device}"
    )

    self_test(
        device=device
    )


if __name__ == "__main__":
    main()