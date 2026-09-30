#!/usr/bin/env python3
"""
SEN2NEON Proposed Hybrid CNN + Transformer
==========================================

Primary architecture for 12-band 4x multispectral
super-resolution.

Input:
    [B, 12, 64, 64]

Output:
    [B, 12, 256, 256]

Architecture:

                    LR
                    |
          +---------+---------+
          |                   |
          v                   v
      CNN branch       Transformer branch
          |                   |
          v                   v
    Local features       Context features
          |                   |
          +---------+---------+
                    |
                    v
              Feature Fusion
                    |
                    v
             Spectral Fusion
                    |
                    v
                4x SR
                    |
                    v
             Learned residual
                    |
                    +
                Bicubic LR
                    |
                    v
                   HR

The model is designed as the primary architecture to compare
against the independent CNN and Transformer baselines.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_IN_CHANNELS = 12
DEFAULT_OUT_CHANNELS = 12
DEFAULT_FEATURES = 64
DEFAULT_TRANSFORMER_DIM = 64
DEFAULT_CNN_BLOCKS = 6
DEFAULT_TRANSFORMER_BLOCKS = 4
DEFAULT_HEADS = 4
DEFAULT_WINDOW_SIZE = 8
DEFAULT_SCALE = 4


# ============================================================
# CNN RESIDUAL BLOCK
# ============================================================

class ResidualCNNBlock(nn.Module):
    """
    Lightweight residual CNN block.

    Conv -> GELU -> Conv -> residual.
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
            padding=1,
        )

        self.activation = nn.GELU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
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
# WINDOW PARTITION
# ============================================================

def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    [B,H,W,C]
        ->
    [B*num_windows, window_size², C]
    """

    B, H, W, C = x.shape

    if H % window_size != 0:
        raise ValueError(
            f"H={H} must be divisible by "
            f"window_size={window_size}"
        )

    if W % window_size != 0:
        raise ValueError(
            f"W={W} must be divisible by "
            f"window_size={window_size}"
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

    [B*num_windows, window_size², C]
        ->
    [B,H,W,C]
    """

    C = windows.shape[-1]

    num_windows_per_image = (
        height // window_size
    ) * (
        width // window_size
    )

    expected = (
        batch_size
        * num_windows_per_image
    )

    if windows.shape[0] != expected:

        raise ValueError(
            "Window count mismatch."
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
# TRANSFORMER MLP
# ============================================================

class TransformerMLP(nn.Module):
    """
    Feed-forward network used inside transformer blocks.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 4.0,
    ) -> None:

        super().__init__()

        hidden = int(
            dim * expansion
        )

        self.fc1 = nn.Linear(
            dim,
            hidden,
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            hidden,
            dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        return x


# ============================================================
# LOCAL TRANSFORMER BLOCK
# ============================================================

class LocalTransformerBlock(
    nn.Module
):
    """
    Window-based Transformer block.

    No global attention is used.

    Attention is restricted to local windows.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,
    ) -> None:

        super().__init__()

        if dim % heads != 0:

            raise ValueError(
                "dim must be divisible "
                "by heads."
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
                num_heads=heads,
                batch_first=True,
                dropout=0.0,
            )
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.mlp = TransformerMLP(
            dim=dim,
            expansion=4.0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        B, H, W, C = x.shape

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        residual = x

        normalized = self.norm1(
            x
        )

        windows = window_partition(
            normalized,
            self.window_size,
        )

        attention_output, _ = (
            self.attention(
                windows,
                windows,
                windows,
                need_weights=False,
            )
        )

        attention_output = (
            window_reverse(
                attention_output,
                self.window_size,
                H,
                W,
                B,
            )
        )

        x = (
            residual
            + attention_output
        )

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
# SPECTRAL FUSION
# ============================================================

class SpectralFusion(nn.Module):
    """
    Cross-spectral feature fusion.

    Uses 1x1 convolutions so information from different
    spectral channels can interact without changing
    spatial resolution.

    This is important because Sentinel-2 bands are not
    independent images.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.activation = nn.GELU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.norm(x)

        x = self.conv1(x)

        x = self.activation(x)

        x = self.conv2(x)

        return (
            residual + x
        )


# ============================================================
# UPSAMPLING
# ============================================================

class UpsampleBlock(nn.Module):
    """
    2x PixelShuffle block.
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
# CNN BRANCH
# ============================================================

class CNNBranch(nn.Module):
    """
    Local spatial feature branch.
    """

    def __init__(
        self,
        in_channels: int,
        features: int,
        blocks: int,
    ) -> None:

        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.Sequential(
            *[
                ResidualCNNBlock(
                    features
                )
                for _ in range(blocks)
            ]
        )

        self.tail = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.head(x)

        residual = x

        x = self.blocks(x)

        x = self.tail(x)

        return (
            x + residual
        )


# ============================================================
# TRANSFORMER BRANCH
# ============================================================

class TransformerBranch(nn.Module):
    """
    Local window-attention branch.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        heads: int,
        window_size: int,
    ) -> None:

        super().__init__()

        self.embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList(
            [
                LocalTransformerBlock(
                    dim=embed_dim,
                    heads=heads,
                    window_size=window_size,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.output = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.embed(x)

        residual = x

        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        for block in self.blocks:

            x = block(x)

        x = self.norm(x)

        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        x = self.output(x)

        return (
            x + residual
        )


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridCNNTransformerSR(
    nn.Module
):
    """
    Proposed CNN + Transformer hybrid.

    The CNN and Transformer branches process the same
    multispectral input independently and are then fused.

    The final prediction is a residual over bicubic
    interpolation.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        features: int = DEFAULT_FEATURES,
        transformer_dim: int = DEFAULT_TRANSFORMER_DIM,
        cnn_blocks: int = DEFAULT_CNN_BLOCKS,
        transformer_blocks: int = DEFAULT_TRANSFORMER_BLOCKS,
        heads: int = DEFAULT_HEADS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        scale: int = DEFAULT_SCALE,
    ) -> None:

        super().__init__()

        if scale != 4:

            raise ValueError(
                "Only 4x super-resolution "
                "is currently supported."
            )

        if features != transformer_dim:

            raise ValueError(
                "features and transformer_dim "
                "must be equal for direct "
                "branch fusion."
            )

        if features % heads != 0:

            raise ValueError(
                "features must be divisible "
                "by number of attention heads."
            )

        self.in_channels = (
            in_channels
        )

        self.out_channels = (
            out_channels
        )

        self.window_size = (
            window_size
        )

        # ----------------------------------------------------
        # Two feature branches
        # ----------------------------------------------------

        self.cnn_branch = CNNBranch(
            in_channels=in_channels,
            features=features,
            blocks=cnn_blocks,
        )

        self.transformer_branch = (
            TransformerBranch(
                in_channels=in_channels,
                embed_dim=transformer_dim,
                depth=transformer_blocks,
                heads=heads,
                window_size=window_size,
            )
        )

        # ----------------------------------------------------
        # Branch fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(
                features * 2,
                features,
                kernel_size=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),
        )

        # ----------------------------------------------------
        # Spectral feature fusion
        # ----------------------------------------------------

        self.spectral_fusion = (
            SpectralFusion(
                features
            )
        )

        # ----------------------------------------------------
        # Additional refinement
        # ----------------------------------------------------

        self.refinement = nn.Sequential(

            ResidualCNNBlock(
                features
            ),

            ResidualCNNBlock(
                features
            ),
        )

        # ----------------------------------------------------
        # Learned 4x upsampling
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            UpsampleBlock(
                features
            ),

            UpsampleBlock(
                features
            ),
        )

        # ----------------------------------------------------
        # Output reconstruction
        # ----------------------------------------------------

        self.reconstruction = (
            nn.Conv2d(
                features,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        )

        # ----------------------------------------------------
        # Initialize
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
                f"Input dimensions must be "
                f"divisible by "
                f"window_size={self.window_size}. "
                f"Got {H}x{W}."
            )

        # ----------------------------------------------------
        # Bicubic reference
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # CNN branch
        # ----------------------------------------------------

        cnn_features = (
            self.cnn_branch(x)
        )

        # ----------------------------------------------------
        # Transformer branch
        # ----------------------------------------------------

        transformer_features = (
            self.transformer_branch(x)
        )

        # ----------------------------------------------------
        # Concatenate both representations
        # ----------------------------------------------------

        fused = torch.cat(
            [
                cnn_features,
                transformer_features,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        fused = self.fusion(
            fused
        )

        # ----------------------------------------------------
        # Spectral fusion
        # ----------------------------------------------------

        fused = (
            self.spectral_fusion(
                fused
            )
        )

        # ----------------------------------------------------
        # Spatial refinement
        # ----------------------------------------------------

        fused = self.refinement(
            fused
        )

        # ----------------------------------------------------
        # 4x upsampling
        # ----------------------------------------------------

        fused = self.upsample(
            fused
        )

        # ----------------------------------------------------
        # Predict residual
        # ----------------------------------------------------

        residual = (
            self.reconstruction(
                fused
            )
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

def create_proposed_model(
    in_channels: int = 12,
    out_channels: int = 12,
    features: int = 64,
    transformer_dim: int = 64,
    cnn_blocks: int = 6,
    transformer_blocks: int = 4,
    heads: int = 4,
    window_size: int = 8,
    scale: int = 4,
) -> HybridCNNTransformerSR:

    return HybridCNNTransformerSR(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        transformer_dim=transformer_dim,
        cnn_blocks=cnn_blocks,
        transformer_blocks=transformer_blocks,
        heads=heads,
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
    print("PROPOSED CNN + TRANSFORMER SELF-TEST")
    print("=" * 72)

    model = create_proposed_model()

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------

    x = torch.rand(
        2,
        12,
        64,
        64,
        device=device,
    )

    with torch.no_grad():

        y = model(x)

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
    # Finite output
    # --------------------------------------------------------

    if not torch.isfinite(y).all():

        raise RuntimeError(
            "Model output contains "
            "NaN or Inf."
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
    # Branch tests
    # --------------------------------------------------------

    with torch.no_grad():

        cnn_features = (
            model.cnn_branch(x)
        )

        transformer_features = (
            model.transformer_branch(x)
        )

    expected_feature_shape = (
        2,
        DEFAULT_FEATURES,
        64,
        64,
    )

    if (
        tuple(cnn_features.shape)
        != expected_feature_shape
    ):

        raise RuntimeError(
            "CNN branch has wrong shape: "
            f"{tuple(cnn_features.shape)}"
        )

    if (
        tuple(transformer_features.shape)
        != expected_feature_shape
    ):

        raise RuntimeError(
            "Transformer branch has "
            "wrong shape: "
            f"{tuple(transformer_features.shape)}"
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
        f"CNN feature shape    : "
        f"{tuple(cnn_features.shape)}"
    )

    print(
        f"Transformer shape    : "
        f"{tuple(transformer_features.shape)}"
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
        "PASS: CNN branch"
    )

    print(
        "PASS: Transformer branch"
    )

    print(
        "PASS: Feature fusion"
    )

    print(
        "PASS: Spectral fusion"
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
        "STATUS: PROPOSED MODEL READY"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Test the proposed "
            "CNN + Transformer model."
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