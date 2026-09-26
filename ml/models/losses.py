# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — Physics-Aware Composite SAR Loss Functions
Implements:
1. Focal Loss (down-weights background ocean pixels, focuses on ambiguous edge clutter).
2. Tversky / Soft Dice Loss (penalizes False Negatives to prevent missed spills).
3. Boundary Loss (approximates Hausdorff distance along the slick perimeter).
4. CompositeSARLoss (jointly optimizes area, boundary, and focal penalties).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object


class FocalLoss(nn.Module):
    """
    Binary / Multi-class Focal Loss for extreme class imbalance in SAR scenes.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Generalization of Dice loss with asymmetric penalty for False Negatives.
    TL = 1 - (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    Setting beta > alpha forces the model to prioritize recall (avoiding missed slicks).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky


class BoundaryLoss(nn.Module):
    """
    Boundary loss penalizing gradient mismatch along slick contours.
    Computes spatial Laplacian / Sobel filters on predictions vs targets.
    """
    def __init__(self):
        super().__init__()
        # 3x3 Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        if TORCH_AVAILABLE:
            self.laplacian = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        else:
            self.laplacian = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        kernel = self.laplacian.to(logits.device)
        pred_edges = F.conv2d(probs, kernel, padding=1)
        target_edges = F.conv2d(targets, kernel, padding=1)
        return F.mse_loss(pred_edges, target_edges)


class CompositeSARLoss(nn.Module):
    """
    Composite SAR Loss:
    L_total = w_focal * L_focal + w_tversky * L_tversky + w_boundary * L_boundary
    """
    def __init__(
        self,
        w_focal: float = 0.4,
        w_tversky: float = 0.4,
        w_boundary: float = 0.2,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7
    ):
        super().__init__()
        self.w_focal = w_focal
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary

        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_tversky = self.tversky(logits, targets)
        l_boundary = self.boundary(logits, targets)
        return (self.w_focal * l_focal) + (self.w_tversky * l_tversky) + (self.w_boundary * l_boundary)
