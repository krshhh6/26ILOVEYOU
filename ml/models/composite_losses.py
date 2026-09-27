# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Phase 2 Composite Multi-Task Loss Functions
=====================================================================
Implements the multi-task objective for Domain-Adversarial U-Net (DANN):
1. Focal Loss (down-weights easy ocean backgrounds, focuses on hard transitions)
2. Tversky Loss (penalizes False Negatives with beta=0.7 to preserve thin filaments)
3. Boundary Contour Loss (sharpens morphological perimeter of slicks)
4. Domain Adversarial Cross-Entropy Loss (enforces sensor invariance)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation to counteract extreme foreground imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape [B, 1, H, W]
            targets: Binary ground truth masks of shape [B, 1, H, W] (0.0 or 1.0)
        """
        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss with tunable FP (alpha) and FN (beta) penalties.
    For maritime oil slicks, beta=0.7 puts stronger penalty on missing thin slicks
    (False Negatives), while alpha=0.3 tolerates slight boundary dilation.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions per batch item
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)

        tp = (p_flat * t_flat).sum(dim=1)
        fp = (p_flat * (1.0 - t_flat)).sum(dim=1)
        fn = ((1.0 - p_flat) * t_flat).sum(dim=1)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - tversky).mean()


class BoundaryContourLoss(nn.Module):
    """
    Morphological boundary loss to eliminate fuzzy perimeter artifacts.
    Derives ground-truth boundary via morphological dilation - erosion,
    and trains auxiliary boundary head using combined Dice + BCE.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def extract_boundary(mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1-2px perimeter using tensor-level morphological gradient:
        Boundary = Dilation(mask) - Erosion(mask)
        """
        # Ensure mask is float
        m = mask.float()
        # Morphological dilation: max_pool with 3x3 kernel
        dilation = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        # Morphological erosion: 1 - max_pool(1 - m)
        erosion = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
        boundary = torch.clamp(dilation - erosion, 0.0, 1.0)
        return boundary

    def forward(self, boundary_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gt_boundary = self.extract_boundary(targets)
        probs = torch.sigmoid(boundary_logits)
        
        # Boundary Dice
        p_flat = probs.view(probs.size(0), -1)
        t_flat = gt_boundary.view(gt_boundary.size(0), -1)
        intersection = (p_flat * t_flat).sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (p_flat.sum(dim=1) + t_flat.sum(dim=1) + self.eps)
        dice_loss = 1.0 - dice.mean()

        # Boundary BCE
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, gt_boundary)

        return dice_loss + bce_loss


class DomainAdversarialCompositeLoss(nn.Module):
    """
    Master Multi-Task Objective for Plan 2.0 Phase 2:
    L_total = w_focal * L_focal + w_tversky * L_tversky + w_bound * L_bound + w_domain * L_domain
    """
    def __init__(
        self,
        weight_focal: float = 1.0,
        weight_tversky: float = 2.0,
        weight_boundary: float = 1.5,
        weight_domain: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        self.w_focal = weight_focal
        self.w_tversky = weight_tversky
        self.w_bound = weight_boundary
        self.w_domain = weight_domain

        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary_loss = BoundaryContourLoss()
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        mask_targets: torch.Tensor,
        domain_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_outputs: Dictionary with keys 'mask', 'boundary', and optional 'domain'
            mask_targets: Ground truth binary masks [B, 1, H, W]
            domain_targets: Ground truth domain labels [B] (long int, 0..num_domains-1)
        
        Returns:
            Dictionary containing 'total', 'focal', 'tversky', 'boundary', 'domain' losses
        """
        mask_logits = model_outputs["mask"]
        boundary_logits = model_outputs.get("boundary")
        domain_logits = model_outputs.get("domain")

        # 1. Focal Loss
        l_focal = self.focal_loss(mask_logits, mask_targets)

        # 2. Tversky Loss
        l_tversky = self.tversky_loss(mask_logits, mask_targets)

        # 3. Boundary Loss
        if boundary_logits is not None:
            l_boundary = self.boundary_loss(boundary_logits, mask_targets)
        else:
            l_boundary = torch.tensor(0.0, device=mask_logits.device)

        # 4. Domain Adversarial Loss
        if domain_logits is not None and domain_targets is not None:
            l_domain = self.domain_loss(domain_logits, domain_targets)
        else:
            l_domain = torch.tensor(0.0, device=mask_logits.device)

        # Total Composite Weighted Sum
        l_total = (
            self.w_focal * l_focal
            + self.w_tversky * l_tversky
            + self.w_bound * l_boundary
            + self.w_domain * l_domain
        )

        return {
            "total": l_total,
            "focal": l_focal,
            "tversky": l_tversky,
            "boundary": l_boundary,
            "domain": l_domain
        }
