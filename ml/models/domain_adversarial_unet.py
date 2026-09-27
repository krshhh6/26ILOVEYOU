# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Domain-Adversarial Multi-Scale U-Net (DANN)
=====================================================================
Implements ML Improvement Plan 2.0 Phase 2 core architecture:
1. Multi-scale feature encoder with Depthwise-Separable residual blocks
2. Atrous Spatial Pyramid Pooling (ASPP) bottleneck with dilations [1, 6, 12, 18]
3. Convolutional Block Attention Module (CBAM) skip-connection gating
4. Gradient Reversal Layer (GRL) & Domain Discriminator for sensor invariance
5. Dual task output heads: Dense Segmentation Mask + Auxiliary Boundary Contour
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object
    class Function:
        pass

try:
    from .aspp import AtrousSpatialPyramidPooling
    from .unet import CBAM
except ImportError:
    from ml.models.aspp import AtrousSpatialPyramidPooling
    from ml.models.unet import CBAM


# ═══════════════════════════════════════════════════════════════
# 1. Gradient Reversal Layer (GRL) for Minimax Domain Invariance
# ═══════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_p: float):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse the gradient sign and scale by adaptation weight lambda_p
        return grad_output.neg() * ctx.lambda_p, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).
    Passes activations forward unchanged, but multiplies gradients by -lambda_p during backprop.
    Forces the feature extractor to learn representations that the domain discriminator cannot separate.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lambda_p: float = 1.0) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_p)


# ═══════════════════════════════════════════════════════════════
# 2. Domain Discriminator Sub-Network
# ═══════════════════════════════════════════════════════════════

class DomainDiscriminator(nn.Module):
    """
    Predicts which dataset / sensor the scene originated from:
    0: Sentinel-1 Dual-Pol (Zenodo)
    1: Krestenitis SOS (Sentinel-1 Single-Pol)
    2: Radarsat / ENVISAT Historical
    3: Web / Google Search sRGB
    """
    def __init__(self, in_channels: int = 256, num_domains: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = self.gap(x).flatten(1)
        max_pool = self.gmp(x).flatten(1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        return self.mlp(features)


# ═══════════════════════════════════════════════════════════════
# 3. Multi-Scale Attention Convolutional Blocks
# ═══════════════════════════════════════════════════════════════

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with BatchNorm and ReLU."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        return x


class ResidualEncoderStage(nn.Module):
    """Encoder stage combining depthwise-separable convs with residual connection."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_c, out_c, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_c, out_c, stride=1)
        
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + res)


class DecoderStage(nn.Module):
    """Decoder stage with bilinear upsampling, CBAM skip-fusion, and convolution."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.cbam = CBAM(skip_c)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_c + skip_c, out_c),
            DepthwiseSeparableConv(out_c, out_c)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Bilinear upsample
        x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Apply CBAM attention to skip connection to suppress coastline / ocean clutter
        skip_attended = self.cbam(skip)
        cat = torch.cat([x_up, skip_attended], dim=1)
        return self.conv(cat)


# ═══════════════════════════════════════════════════════════════
# 4. Master Domain-Adversarial U-Net (DANN-UNet)
# ═══════════════════════════════════════════════════════════════

class DomainAdversarialUNet(nn.Module):
    """
    Plan 2.0 Master Segmentation Network.
    
    Inputs:
        x: [B, 2, 512, 512] (Normalized VV + VH / Pseudo-VH)
        lambda_p: float (Adaptation schedule weight, 0.0 to 1.0)
        
    Outputs (Training Mode):
        dict containing:
            - 'mask': [B, 1, 512, 512] (Oil spill segmentation logits)
            - 'boundary': [B, 1, 512, 512] (Auxiliary boundary contour logits)
            - 'domain': [B, num_domains] (Adversarial domain classification logits)
            
    Outputs (Export / Inference Mode):
        [B, 1, 512, 512] segmentation logits
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        num_domains: int = 4,
        features: tuple = (32, 64, 128, 256),
        export_mode: bool = False
    ):
        super().__init__()
        self.export_mode = export_mode
        self.num_domains = num_domains

        # Initial Stem Conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )

        # 4-Stage Encoder
        self.enc1 = ResidualEncoderStage(features[0], features[0], stride=1) # 512x512
        self.enc2 = ResidualEncoderStage(features[0], features[1], stride=2) # 256x256
        self.enc3 = ResidualEncoderStage(features[1], features[2], stride=2) # 128x128
        self.enc4 = ResidualEncoderStage(features[2], features[3], stride=2) # 64x64

        # Multi-Scale Bottleneck (ASPP)
        self.aspp = AtrousSpatialPyramidPooling(
            in_channels=features[3],
            out_channels=features[3],
            dilations=(1, 6, 12, 18)
        )

        # Adversarial Domain Path (GRL + Discriminator)
        self.grl = GradientReversalLayer()
        self.domain_discriminator = DomainDiscriminator(
            in_channels=features[3],
            num_domains=num_domains
        )

        # 3-Stage Decoder with CBAM Skip-Fusion
        self.dec3 = DecoderStage(in_c=features[3], skip_c=features[2], out_c=features[2]) # 128x128
        self.dec2 = DecoderStage(in_c=features[2], skip_c=features[1], out_c=features[1]) # 256x256
        self.dec1 = DecoderStage(in_c=features[1], skip_c=features[0], out_c=features[0]) # 512x512

        # Dual Output Prediction Heads
        self.seg_head = nn.Conv2d(features[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        lambda_p: float = 1.0
    ) -> Any:
        # Encoder forward pass
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # ASPP Bottleneck multi-scale feature representation
        bottleneck = self.aspp(s4)

        # In pure inference / ONNX export mode, skip domain discriminator
        if self.export_mode:
            d3 = self.dec3(bottleneck, s3)
            d2 = self.dec2(d3, s2)
            d1 = self.dec1(d2, s1)
            return self.seg_head(d1)

        # Adversarial Domain Classification path via Gradient Reversal
        grl_features = self.grl(bottleneck, lambda_p)
        domain_logits = self.domain_discriminator(grl_features)

        # Task Segmentation Decoder path
        d3 = self.dec3(bottleneck, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        mask_logits = self.seg_head(d1)
        boundary_logits = self.boundary_head(d1)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "domain": domain_logits
        }
