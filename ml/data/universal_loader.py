# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Multi-Domain Universal Dataset Loader
===============================================================
Implements ML Improvement Plan 2.0 multi-dataset federation:
- Ingests Sentinel-1 (Zenodo & SOS), Radarsat, and Web/Google Search images
- Applies UniversalSARPreprocessor for radiometric invariance
- Yields domain labels for Domain-Adversarial Neural Network (DANN) training
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset: pass

from ml.data.cdf_normalizer import UniversalSARPreprocessor, PreprocessedSARTensor

# Standard Domain IDs for Domain-Adversarial Training
DOMAIN_IDS = {
    "sentinel1_dualpol": 0,
    "krestenitis_sos": 1,
    "radarsat_envisat": 2,
    "web_search_srgb": 3
}


class UniversalSARDataset(Dataset):
    """
    Unified multi-dataset loader that seamlessly feeds normalized 2-channel tensors
    and ground-truth binary masks to PyTorch models across heterogeneous data sources.
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        return_domain_labels: bool = True
    ):
        """
        Args:
            samples: List of dicts, each with keys:
                - 'image_path': Path to GeoTIFF or PNG/JPEG
                - 'mask_path': Optional Path to binary ground truth mask
                - 'domain': str (e.g. 'sentinel1_dualpol', 'web_search_srgb')
            target_size: (H, W) for input tensors
            augment: Whether to apply spatial augmentations (flips, rotations)
            return_domain_labels: If True, returns domain_id for DANN training
        """
        self.samples = samples
        self.target_size = target_size
        self.augment = augment
        self.return_domain_labels = return_domain_labels
        self.preprocessor = UniversalSARPreprocessor(target_size=target_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        if p.suffix.lower() in ('.tif', '.tiff'):
            try:
                import tifffile
                arr = tifffile.imread(str(p))
                return arr
            except Exception:
                pass
        
        # Standard web image (PNG / JPEG)
        with Image.open(str(p)) as img:
            arr = np.array(img.convert('RGB'))
            return arr

    def _load_mask(self, path: Optional[Union[str, Path]], shape: Tuple[int, int]) -> np.ndarray:
        if path is None:
            return np.zeros(shape, dtype=np.uint8)
        p = Path(path)
        if not p.exists():
            return np.zeros(shape, dtype=np.uint8)

        if p.suffix.lower() in ('.tif', '.tiff'):
            try:
                import tifffile
                mask = tifffile.imread(str(p))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                return (mask > 0).astype(np.uint8)
            except Exception:
                pass

        with Image.open(str(p)) as img:
            mask = np.array(img.convert('L'))
            return (mask > 20).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]
        img_arr = self._load_image(sample_meta['image_path'])
        
        # Universal radiometric preprocessing & 2D swath detrending
        is_tiff_db = Path(sample_meta['image_path']).suffix.lower() in ('.tif', '.tiff')
        prep: PreprocessedSARTensor = self.preprocessor.process_array(
            img_arr, 
            is_direct_geotiff_db=(is_tiff_db and img_arr.dtype in (np.float32, np.float64))
        )
        
        vv = prep.vv
        vh = prep.vh
        
        # Load mask
        mask_path = sample_meta.get('mask_path')
        mask = self._load_mask(mask_path, vv.shape)
        
        # Resize to target_size if needed
        th, tw = self.target_size
        if vv.shape != (th, tw):
            # Bilinear resize for inputs
            vv_pil = Image.fromarray((vv * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
            vh_pil = Image.fromarray((vh * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
            mask_pil = Image.fromarray(mask * 255).resize((tw, th), Image.NEAREST)
            
            vv = np.array(vv_pil, dtype=np.float32) / 255.0
            vh = np.array(vh_pil, dtype=np.float32) / 255.0
            mask = (np.array(mask_pil) > 128).astype(np.uint8)

        # Spatial data augmentations (flips & rotations)
        if self.augment:
            if np.random.rand() > 0.5:  # Horizontal flip
                vv = np.fliplr(vv).copy()
                vh = np.fliplr(vh).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:  # Vertical flip
                vv = np.flipud(vv).copy()
                vh = np.flipud(vh).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)  # Random 90 deg rotation
            if k > 0:
                vv = np.rot90(vv, k).copy()
                vh = np.rot90(vh, k).copy()
                mask = np.rot90(mask, k).copy()

        # Stack into [2, H, W]
        tensor = np.stack([vv, vh], axis=0).astype(np.float32)
        mask_tensor = mask[np.newaxis, :, :].astype(np.float32)
        
        domain_name = sample_meta.get('domain', 'sentinel1_dualpol')
        domain_id = DOMAIN_IDS.get(domain_name, 0)
        
        if TORCH_AVAILABLE:
            return {
                "input": torch.from_numpy(tensor),
                "mask": torch.from_numpy(mask_tensor),
                "domain_id": torch.tensor(domain_id, dtype=torch.long),
                "name": Path(sample_meta['image_path']).stem
            }
        else:
            return {
                "input": tensor,
                "mask": mask_tensor,
                "domain_id": domain_id,
                "name": Path(sample_meta['image_path']).stem
            }
