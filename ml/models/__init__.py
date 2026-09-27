# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Neural Network Models Package
=======================================================
Exports all segmentation architectures, attention mechanisms,
domain-adversarial components, classifiers, and loss functions.
"""

from ml.models.aspp import AtrousSpatialPyramidPooling
from ml.models.domain_adversarial_unet import (
    DomainAdversarialUNet,
    GradientReversalLayer,
    DomainDiscriminator,
    DepthwiseSeparableConv,
    ResidualEncoderStage,
    DecoderStage
)
from ml.models.composite_losses import (
    DomainAdversarialCompositeLoss,
    BinaryFocalLoss,
    TverskyLoss,
    BoundaryContourLoss
)
from ml.models.unet import CompactSARUNet, UNetS1SAR, CBAM
from ml.models.losses import FocalLoss, BoundaryLoss, CompositeSARLoss
from ml.models.dual_pol_net import (
    DualPolOilSpillNet,
    SpillSegNet,
    get_classifier,
    get_segmenter,
    get_model,
    count_params,
    SqueezeExcitation,
    DepthwiseSeparableBlock,
    ConvBlock,
    DSConvBlock
)

__all__ = [
    # Phase 2 DANN & ASPP
    "AtrousSpatialPyramidPooling",
    "DomainAdversarialUNet",
    "GradientReversalLayer",
    "DomainDiscriminator",
    "DepthwiseSeparableConv",
    "ResidualEncoderStage",
    "DecoderStage",
    "DomainAdversarialCompositeLoss",
    "BinaryFocalLoss",
    "TverskyLoss",
    "BoundaryContourLoss",
    # Compact & Baseline UNets
    "CompactSARUNet",
    "UNetS1SAR",
    "CBAM",
    "FocalLoss",
    "BoundaryLoss",
    "CompositeSARLoss",
    # Legacy Dual-Pol Architectures
    "DualPolOilSpillNet",
    "SpillSegNet",
    "get_classifier",
    "get_segmenter",
    "get_model",
    "count_params",
    "SqueezeExcitation",
    "DepthwiseSeparableBlock",
    "ConvBlock",
    "DSConvBlock",
]
