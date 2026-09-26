# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Universal SAR Preprocessor Service
=============================================================
Provides production backend integration for ML Improvement Plan 2.0:
- 2D Swath Background Polynomial Detrending
- Adaptive Marine Cumulative Distribution Function (CDF) Specification
- Zero-Shot Format Invariance (GeoTIFF, Web PNG/JPEG, Colorized SAR)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import io


class UniversalSARPreprocessorService:
    """
    Backend service delivering universal radiometric invariance across
    Sentinel-1, Radarsat, and compressed 8-bit web SAR pictures.
    """
    def __init__(self, canonical_ocean_level: float = 0.62):
        self.canonical_ocean_level = canonical_ocean_level

    def detrend_2d_swath(
        self,
        image_2d: np.ndarray,
        marine_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtracts 2D illumination / antenna gain slope across the scene.
        """
        h, w = image_2d.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        step = max(1, min(h, w) // 64)
        sub_y = y_coords[::step, ::step].ravel()
        sub_x = x_coords[::step, ::step].ravel()
        sub_mask = marine_mask[::step, ::step].ravel()
        sub_vals = image_2d[::step, ::step].ravel()

        valid_idx = sub_mask & np.isfinite(sub_vals)
        if valid_idx.sum() < 30:
            return image_2d.copy(), np.full_like(image_2d, float(np.median(image_2d)))

        px = sub_x[valid_idx]
        py = sub_y[valid_idx]
        pz = sub_vals[valid_idx].astype(np.float64)

        marine_median = float(np.median(pz))
        p20, p80 = np.percentile(pz, 20), np.percentile(pz, 80)
        bg_idx = (pz >= p20) & (pz <= p80)
        if bg_idx.sum() > 20:
            px, py, pz = px[bg_idx], py[bg_idx], pz[bg_idx]

        A = np.column_stack([px, py, np.ones_like(px)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
            fitted_surface = (coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]).astype(np.float32)
        except Exception:
            fitted_surface = np.full_like(image_2d, marine_median, dtype=np.float32)

        detrended = image_2d.astype(np.float32) - fitted_surface + marine_median
        return detrended, fitted_surface

    def process_image_bytes(
        self, 
        image_bytes: bytes,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes arbitrary image bytes into normalized [2, target_H, target_W] float32 tensor.
        """
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert('RGB')
            if rgb.size != (target_size[1], target_size[0]):
                rgb = rgb.resize((target_size[1], target_size[0]), Image.BILINEAR)
            arr = np.array(rgb)

        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)

        # Marine range: [12, 165]
        marine_mask = (gray >= 12.0) & (gray <= 165.0)
        land_mask = (gray > 165.0).astype(np.uint8)

        detrended, fitted_surface = self.detrend_2d_swath(gray, marine_mask)
        marine_vals = detrended[marine_mask]
        ambient_ocean = float(np.median(marine_vals)) if len(marine_vals) > 50 else 85.0

        vv_norm = np.zeros_like(detrended, dtype=np.float32)
        lower_mask = marine_mask & (detrended <= ambient_ocean)
        ratio_lower = (detrended[lower_mask] - 12.0) / max(1.0, ambient_ocean - 12.0)
        vv_norm[lower_mask] = 0.10 + 0.52 * np.clip(ratio_lower, 0.0, 1.0)

        upper_mask = marine_mask & (detrended > ambient_ocean)
        ratio_upper = (detrended[upper_mask] - ambient_ocean) / max(1.0, 165.0 - ambient_ocean)
        vv_norm[upper_mask] = 0.62 + 0.30 * np.clip(ratio_upper, 0.0, 1.0)

        vv_norm[gray > 165.0] = 1.0
        vv_norm[gray < 12.0] = self.canonical_ocean_level

        # Synthesize pseudo-VH
        base_vh = vv_norm - 0.05
        slick_suppression = np.clip((self.canonical_ocean_level - vv_norm) / self.canonical_ocean_level, 0.0, 1.0)
        vh_norm = np.clip(base_vh + 0.04 * slick_suppression, 0.0, 1.0)

        tensor = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)

        meta = {
            "ambient_ocean_mean": ambient_ocean,
            "surface_gradient_delta": float(fitted_surface.max() - fitted_surface.min()),
            "land_fraction": float(land_mask.mean()),
            "is_calibrated": True
        }
        return tensor, meta


universal_sar_service = UniversalSARPreprocessorService()
