# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Universal Radiometric Invariance & 2D Detrending Engine
=================================================================================
Implements ML Improvement Plan 2.0 components:
1. Adaptive Marine Cumulative Distribution Function (CDF) Normalization
2. 2D Swath Polynomial Background Detrending (low-wind & antenna roll-off suppression)
3. Physics-Derived Pseudo-VH Cross-Polarization Synthesizer (single-band & web inputs)
4. Land & Metallic Structure Masking
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PreprocessedSARTensor:
    vv: np.ndarray             # Normalized VV channel [H, W], float32 in [0.0, 1.0]
    vh: np.ndarray             # Normalized VH channel [H, W], float32 in [0.0, 1.0]
    land_mask: np.ndarray      # Binary mask of terrestrial land/vessel [H, W], uint8
    detrended_surface: np.ndarray # The fitted 2D background gradient [H, W], float32
    ambient_ocean_mean: float  # Baseline calibrated marine reflectance
    is_web_srgb: bool          # Whether input was 8-bit web image vs 16-bit GeoTIFF


def swath_2d_detrend(
    image: np.ndarray, 
    valid_marine_mask: np.ndarray,
    order: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a 2D surface to the large-scale marine background and subtracts it.
    Completely eliminates the 'dark edge' low-wind / antenna incidence angle trap
    while preserving sharp, localized oil slick filaments.
    
    Args:
        image: 2D array [H, W] (float32 or uint8)
        valid_marine_mask: 2D boolean array where True = open water (not land or void)
        order: Polynomial order (1 for planar tilt, 2 for quadratic swath roll-off)
        
    Returns:
        detrended_image: Flattened marine image [H, W] centered around ocean median
        fitted_surface: The background illumination surface [H, W]
    """
    h, w = image.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Sample points on the valid marine surface (subsampled for speed)
    step = max(1, min(h, w) // 64)
    sub_y = y_coords[::step, ::step].ravel()
    sub_x = x_coords[::step, ::step].ravel()
    sub_mask = valid_marine_mask[::step, ::step].ravel()
    sub_vals = image[::step, ::step].ravel()
    
    valid_idx = sub_mask & np.isfinite(sub_vals)
    if valid_idx.sum() < 30:
        # Not enough marine pixels to fit surface; return original
        return image.copy(), np.full_like(image, np.median(image))
        
    px = sub_x[valid_idx]
    py = sub_y[valid_idx]
    pz = sub_vals[valid_idx].astype(np.float64)
    
    # Robust median to resist outlier slicks
    marine_median = float(np.median(pz))
    
    # Exclude extreme anomalies (dense oil slicks < 25th percentile or land > 85th) during surface fitting
    p20, p80 = np.percentile(pz, 20), np.percentile(pz, 80)
    bg_idx = (pz >= p20) & (pz <= p80)
    if bg_idx.sum() > 20:
        px = px[bg_idx]
        py = py[bg_idx]
        pz = pz[bg_idx]
        
    # Build design matrix for 2D polynomial
    if order == 1:
        # Planar tilt: z = a*x + b*y + c
        A = np.column_stack([px, py, np.ones_like(px)])
    else:
        # Quadratic curvature: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
        A = np.column_stack([px**2, py**2, px*py, px, py, np.ones_like(px)])
        
    # Solve least-squares for surface coefficients
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
        if order == 1:
            fitted_surface = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]
        else:
            fitted_surface = (
                coeffs[0] * x_coords**2 + coeffs[1] * y_coords**2 +
                coeffs[2] * x_coords * y_coords + coeffs[3] * x_coords +
                coeffs[4] * y_coords + coeffs[5]
            )
    except Exception:
        fitted_surface = np.full_like(image, marine_median, dtype=np.float32)
        
    fitted_surface = fitted_surface.astype(np.float32)
    
    # Detrend: subtract fitted gradient, restore global marine median
    detrended = image.astype(np.float32) - fitted_surface + marine_median
    return detrended, fitted_surface


def synthesize_pseudo_vh(vv_normalized: np.ndarray, ambient_ocean_level: float = 0.62) -> np.ndarray:
    """
    Synthesizes a physically-realistic cross-polarization (VH) channel from co-pol (VV).
    In real ocean SAR:
    - Oil dampens VV much more aggressively than VH.
    - Low-wind areas have very low backscatter in VV, but VH stays near the radar noise floor.
    - Land and ship structures exhibit intense volume/double-bounce scattering in both channels.
    """
    # Base VH is slightly lower than VV by physical cross-pol offset (~0.05 in normalized space)
    base_vh = vv_normalized - 0.05
    
    # In severe capillary damping regions (oil slicks), the polarimetric ratio diverges:
    # VV drops sharply, while VH levels off at the system noise floor
    slick_suppression = np.clip((ambient_ocean_level - vv_normalized) / max(0.01, ambient_ocean_level), 0.0, 1.0)
    vh_synth = base_vh + 0.04 * slick_suppression
    
    return np.clip(vh_synth, 0.0, 1.0).astype(np.float32)


class UniversalSARPreprocessor:
    """
    Production-grade universal preprocessor that maps any SAR image
    (16-bit GeoTIFF, 8-bit web screenshot, single-channel or dual-pol)
    into the canonical [2, H, W] tensor expected by the deep learning models.
    """
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.canonical_ocean_level = 0.62  # Matches Zenodo & SOS training distribution
        self.canonical_oil_core = 0.22     # Canonical core damping level

    def process_array(
        self,
        image_data: np.ndarray,
        is_direct_geotiff_db: bool = False
    ) -> PreprocessedSARTensor:
        """
        Processes a 2D or 3D numpy image array.
        
        Args:
            image_data: 2D [H, W] grayscale, or 3D [H, W, 2]/[2, H, W] dual-pol, or 3D [H, W, 3] RGB
            is_direct_geotiff_db: If True, indicates array contains raw float Sigma0 dB values
        """
        # 1. Normalize dimensions
        if image_data.ndim == 3:
            if image_data.shape[0] == 2:  # [2, H, W]
                raw_vv = image_data[0].astype(np.float32)
                raw_vh = image_data[1].astype(np.float32)
                has_vh = True
            elif image_data.shape[-1] == 2:  # [H, W, 2]
                raw_vh = image_data[:, :, 0].astype(np.float32)
                raw_vv = image_data[:, :, 1].astype(np.float32)
                has_vh = True
            elif image_data.shape[-1] in (3, 4):  # RGB / RGBA web image
                raw_vv = (
                    0.299 * image_data[:, :, 0] +
                    0.587 * image_data[:, :, 1] +
                    0.114 * image_data[:, :, 2]
                ).astype(np.float32)
                raw_vh = None
                has_vh = False
            else:
                raw_vv = image_data[:, :, 0].astype(np.float32)
                raw_vh = None
                has_vh = False
        elif image_data.ndim == 2:
            raw_vv = image_data.astype(np.float32)
            raw_vh = None
            has_vh = False
        else:
            raise ValueError(f"Unsupported image dimension {image_data.ndim}")

        h, w = raw_vv.shape

        # 2. Distinguish raw float dB from 8-bit web imagery
        is_web = not is_direct_geotiff_db and (np.nanmax(raw_vv) > 1.0 or raw_vv.dtype == np.uint8)

        if is_web:
            # 8-bit Web / Screenshot Image Mode
            # Valid marine range: [12, 165]
            marine_mask = (raw_vv >= 12.0) & (raw_vv <= 165.0)
            land_mask = (raw_vv > 165.0).astype(np.uint8)
            
            # Detrend 2D background gradient
            detrended_vv, fitted_surface = swath_2d_detrend(raw_vv, marine_mask, order=1)
            
            # Recalculate marine median on detrended water
            marine_pixels = detrended_vv[marine_mask]
            ambient_ocean = float(np.median(marine_pixels)) if len(marine_pixels) > 50 else 85.0
            
            # Adaptive CDF specification: map [12, ambient] -> [0.10, 0.62]
            vv_norm = np.zeros_like(detrended_vv, dtype=np.float32)
            
            # Below ambient ocean (capillary wave damping / slicks)
            lower_mask = marine_mask & (detrended_vv <= ambient_ocean)
            ratio_lower = (detrended_vv[lower_mask] - 12.0) / max(1.0, ambient_ocean - 12.0)
            vv_norm[lower_mask] = 0.10 + 0.52 * np.clip(ratio_lower, 0.0, 1.0)
            
            # Above ambient ocean (rough water / clutter)
            upper_mask = marine_mask & (detrended_vv > ambient_ocean)
            ratio_upper = (detrended_vv[upper_mask] - ambient_ocean) / max(1.0, 165.0 - ambient_ocean)
            vv_norm[upper_mask] = 0.62 + 0.30 * np.clip(ratio_upper, 0.0, 1.0)
            
            # Terrestrial land & NoData void
            vv_norm[raw_vv > 165.0] = 1.0
            vv_norm[raw_vv < 12.0] = self.canonical_ocean_level
            
            vh_norm = synthesize_pseudo_vh(vv_norm, self.canonical_ocean_level) if not has_vh else np.clip(raw_vh / 255.0, 0.0, 1.0)

        else:
            # Calibrated Scientific GeoTIFF Mode (Sigma0 dB)
            # Standard Sentinel-1 bounds: VV in [-32.0, -10.0] dB, VH in [-42.0, -20.0] dB
            vv_min, vv_max = -32.0, -10.0
            vh_min, vh_max = -42.0, -20.0
            
            marine_mask = (raw_vv > -50.0) & (raw_vv < 5.0)
            land_mask = (raw_vv >= -8.0).astype(np.uint8)
            
            detrended_vv, fitted_surface = swath_2d_detrend(raw_vv, marine_mask, order=1)
            ambient_ocean = float(np.median(detrended_vv[marine_mask])) if marine_mask.sum() > 50 else -18.0
            
            vv_clipped = np.clip(detrended_vv, vv_min, vv_max)
            vv_norm = (vv_clipped - vv_min) / (vv_max - vv_min)
            
            if has_vh and raw_vh is not None:
                vh_clipped = np.clip(raw_vh, vh_min, vh_max)
                vh_norm = (vh_clipped - vh_min) / (vh_max - vh_min)
            else:
                vh_norm = synthesize_pseudo_vh(vv_norm, self.canonical_ocean_level)

        return PreprocessedSARTensor(
            vv=vv_norm.astype(np.float32),
            vh=vh_norm.astype(np.float32),
            land_mask=land_mask,
            detrended_surface=fitted_surface,
            ambient_ocean_mean=ambient_ocean,
            is_web_srgb=is_web
        )
