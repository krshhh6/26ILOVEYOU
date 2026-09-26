# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — SAR Transforms & Radiometric Augmentation
Provides physics-aware transformations for Sentinel-1 C-Band SAR imagery:
1. Radiometric dB normalization [VV & VH bands].
2. Multiplicative Rayleigh/Gamma speckle noise augmentation.
3. Adaptive Lee speckle filtering.
4. Rigid spatial geometric transformations (flips, 90-degree rotations).
"""

import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple, Optional


def normalize_sar_band(
    arr_db: np.ndarray,
    min_db: float = -32.0,
    max_db: float = -4.0
) -> np.ndarray:
    """
    Normalizes a SAR backscatter array in decibels (dB) to [0.0, 1.0].
    Values outside [min_db, max_db] are smoothly clipped.
    """
    clipped = np.clip(arr_db, min_db, max_db)
    norm = (clipped - min_db) / (max_db - min_db)
    return norm.astype(np.float32)


def add_synthetic_speckle(
    image: np.ndarray,
    num_looks: int = 4
) -> np.ndarray:
    """
    Adds multiplicative Gamma speckle noise to simulate low-look SAR acquisitions.
    Intensity I is modulated by Gamma(L, 1/L) where L is the equivalent number of looks.
    """
    shape = image.shape
    # Mean = 1.0, Variance = 1 / num_looks
    noise = np.random.gamma(shape=num_looks, scale=1.0 / num_looks, size=shape).astype(np.float32)
    noisy = image * noise
    return np.clip(noisy, 0.0, 1.0)


def adaptive_lee_filter(
    image: np.ndarray,
    window_size: int = 5,
    num_looks: int = 1
) -> np.ndarray:
    """
    Adaptive Lee Filter for SAR speckle reduction.
    Preserves oil slick edges and damping boundaries while smoothing sea clutter.
    """
    img = image.astype(np.float32)
    local_mean = uniform_filter(img, size=window_size)
    local_sqr_mean = uniform_filter(np.square(img), size=window_size)
    local_var = np.maximum(local_sqr_mean - np.square(local_mean), 1e-6)

    noise_var = 1.0 / max(num_looks, 1)
    weights = local_var / (local_var + noise_var * np.square(local_mean) + 1e-6)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (img - local_mean)
    return filtered.astype(np.float32)


class SARRandomAugment:
    """
    Composes random spatial and radiometric augmentations for SAR training patches.
    """
    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        p_speckle: float = 0.3,
        p_intensity_jitter: float = 0.4
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_speckle = p_speckle
        self.p_intensity_jitter = p_intensity_jitter

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        image: (C, H, W) float32 in [0, 1]
        mask: (H, W) or (C, H, W) uint8 / int64
        """
        # Horizontal flip
        if np.random.rand() < self.p_hflip:
            image = np.flip(image, axis=-1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-1).copy()

        # Vertical flip
        if np.random.rand() < self.p_vflip:
            image = np.flip(image, axis=-2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-2).copy()

        # Random 90 deg rotation (0, 90, 180, 270)
        if np.random.rand() < self.p_rot90:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(-2, -1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(-2, -1)).copy()

        # Radiometric speckle jitter (applied to image channels only)
        if np.random.rand() < self.p_speckle:
            looks = np.random.randint(3, 8)
            for c in range(image.shape[0]):
                image[c] = add_synthetic_speckle(image[c], num_looks=looks)

        # Radiometric scale jitter (simulating slight calibration offset or incidence angle shift)
        if np.random.rand() < self.p_intensity_jitter:
            scale = np.random.uniform(0.92, 1.08)
            image = np.clip(image * scale, 0.0, 1.0)

        return image, mask
