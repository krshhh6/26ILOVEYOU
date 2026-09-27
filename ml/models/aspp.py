# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Atrous Spatial Pyramid Pooling (ASPP)
===============================================================
Implements multi-scale dilated convolutions for ML Improvement Plan 2.0:
- Rates [1, 6, 12, 18] capture both ultra-thin 1-pixel ship bilge wakes
  and massive 5km oil pools without losing spatial resolution.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object


class ASPPConv(nn.Module):
    """Dilated convolution branch with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ASPPPooling(nn.Module):
    """Global average pooling branch for scene-level ambient context."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pool = self.gap(x)
        out = self.relu(self.bn(self.conv(pool)))
        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)


class AtrousSpatialPyramidPooling(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Bottleneck.
    Extracts multi-scale contextual features across 5 parallel branches:
    1. 1x1 standard convolution (local point-wise details)
    2. 3x3 dilated conv with dilation rate 6 (local slick boundaries)
    3. 3x3 dilated conv with dilation rate 12 (medium slick clusters)
    4. 3x3 dilated conv with dilation rate 18 (wide multi-kilometer spills)
    5. Global Average Pooling (entire scene backscatter context)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations: tuple = (1, 6, 12, 18),
        dropout: float = 0.2
    ):
        super().__init__()
        self.branches = nn.ModuleList()

        # Branch 1: 1x1 conv
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # Branches 2-4: 3x3 dilated convs
        for d in dilations[1:]:
            self.branches.append(ASPPConv(in_channels, out_channels, dilation=d))

        # Branch 5: Image-level global pooling
        self.branches.append(ASPPPooling(in_channels, out_channels))

        # Projection back to target feature dimension
        total_channels = out_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        return self.project(concat)
