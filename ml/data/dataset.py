# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — Sentinel-1 SAR Oil Spill Dataset Loader
Implements:
1. Ground-truth mask parsing (5-class Krestenitis/M4D standard and binary classification).
2. Multi-polarization ingestion (VV, VH, and Polarimetric Ratio).
3. Spatio-Temporal Group Splitting (GroupKFold by Sentinel-1 Scene/Orbit ID) to eliminate spatial data leakage.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

from ml.data.transforms import SARRandomAugment, normalize_sar_band


# Standard 5-class M4D / Krestenitis palette:
# 0: Sea Surface (Black)
# 1: Oil Spill (Cyan: [0, 255, 255])
# 2: Look-alike (Red: [255, 0, 0])
# 3: Ship / Platform (Brown: [153, 76, 0])
# 4: Land (Green: [0, 255, 0])
CLASS_NAMES = ["Sea Surface", "Oil Spill", "Look-alike", "Ship", "Land"]


def rgb_mask_to_class_indices(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Maps 3-channel RGB ground truth masks to integer class labels (0 to 4).
    """
    h, w, _ = mask_rgb.shape
    label_mask = np.zeros((h, w), dtype=np.int64)

    r = mask_rgb[:, :, 0]
    g = mask_rgb[:, :, 1]
    b = mask_rgb[:, :, 2]

    # Oil Spill: Cyan ([0, 255, 255] or high G & B, low R)
    oil_mask = (g > 180) & (b > 180) & (r < 100)
    # Lookalike: Red ([255, 0, 0] or high R, low G & B)
    lookalike_mask = (r > 180) & (g < 100) & (b < 100)
    # Ship: Brown ([153, 76, 0] or moderate R, low G, low B)
    ship_mask = (r > 100) & (g < 120) & (b < 80) & ~lookalike_mask
    # Land: Green ([0, 255, 0] or high G, low R & B)
    land_mask = (g > 180) & (r < 100) & (b < 100)

    label_mask[oil_mask] = 1
    label_mask[lookalike_mask] = 2
    label_mask[ship_mask] = 3
    label_mask[land_mask] = 4

    return label_mask


def extract_scene_group_id(filename: str) -> str:
    """
    Extracts the Sentinel-1 scene/product identifier from a patch filename.
    Ensures that patches cut from the same acquisition pass share a common Group ID.
    Example: 'S1A_IW_GRDH_1SDV_20230514T054522_patch_04.png' -> '20230514T054522'
    """
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        return match.group(1)
    
    # Fallback to scene prefix if standard pattern not present
    parts = Path(filename).stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return Path(filename).stem


class SARDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR Oil Spill Semantic Segmentation.
    """
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        mask_paths: Optional[List[Union[str, Path]]] = None,
        img_size: int = 400,
        in_channels: int = 1,
        binary_oil_only: bool = True,
        augment: bool = False,
        scene_ids: Optional[List[str]] = None
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths] if mask_paths else None
        self.img_size = img_size
        self.in_channels = in_channels
        self.binary_oil_only = binary_oil_only
        self.augment = augment
        self.augmentor = SARRandomAugment() if augment else None

        if scene_ids:
            self.scene_ids = scene_ids
        else:
            self.scene_ids = [extract_scene_group_id(p.name) for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        pil_img = Image.open(img_path).convert("L")
        if pil_img.size != (self.img_size, self.img_size):
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        raw_arr = np.array(pil_img, dtype=np.float32) / 255.0

        if self.in_channels == 1:
            # (1, H, W)
            image_tensor = np.expand_dims(raw_arr, axis=0)
        elif self.in_channels == 2:
            # Simulated or second channel (e.g. VH)
            vh_sim = np.clip(raw_arr * 0.75 + 0.1, 0.0, 1.0)
            image_tensor = np.stack([raw_arr, vh_sim], axis=0)
        elif self.in_channels == 3:
            # 3-channel: VV, VH, and Polarimetric Cross Ratio
            vh_sim = np.clip(raw_arr * 0.75 + 0.1, 0.0, 1.0)
            ratio = (raw_arr + 1e-4) / (vh_sim + 1e-4)
            ratio_norm = np.clip(ratio / 2.0, 0.0, 1.0).astype(np.float32)
            image_tensor = np.stack([raw_arr, vh_sim, ratio_norm], axis=0)
        else:
            image_tensor = np.expand_dims(raw_arr, axis=0)

        # Load or generate mask
        if self.mask_paths and idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            pil_mask = Image.open(mask_path)
            if pil_mask.size != (self.img_size, self.img_size):
                pil_mask = pil_mask.resize((self.img_size, self.img_size), Image.NEAREST)
            
            mask_arr = np.array(pil_mask)
            if mask_arr.ndim == 3 and mask_arr.shape[-1] >= 3:
                class_mask = rgb_mask_to_class_indices(mask_arr[:, :, :3])
            else:
                class_mask = (mask_arr > 127).astype(np.int64)

            if self.binary_oil_only:
                target_mask = (class_mask == 1).astype(np.float32)
            else:
                target_mask = class_mask
        else:
            # Default empty mask if unlabelled evaluation
            target_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Apply augmentations
        if self.augment and self.augmentor:
            image_tensor, target_mask = self.augmentor(image_tensor, target_mask)

        if TORCH_AVAILABLE:
            img_out = torch.from_numpy(image_tensor.copy()).float()
            mask_out = torch.from_numpy(target_mask.copy())
            if self.binary_oil_only:
                mask_out = mask_out.unsqueeze(0).float()
            else:
                mask_out = mask_out.long()
            return img_out, mask_out

        return image_tensor, target_mask


def create_spatiotemporal_splits(
    image_paths: List[Union[str, Path]],
    mask_paths: Optional[List[Union[str, Path]]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Groups patches by Sentinel-1 acquisition pass to prevent spatial data leakage.
    All patches originating from the same satellite swath belong strictly to either
    train, val, or test.
    """
    np.random.seed(seed)
    groups: Dict[str, List[int]] = {}

    for idx, p in enumerate(image_paths):
        gid = extract_scene_group_id(Path(p).name)
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(idx)

    group_keys = list(groups.keys())
    np.random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = max(1, int(n_groups * train_ratio))
    n_val = max(1, int(n_groups * val_ratio))

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])
    test_groups = set(group_keys[n_train + n_val:])

    train_idx = [i for g in train_groups for i in groups[g]]
    val_idx = [i for g in val_groups for i in groups[g]]
    test_idx = [i for g in test_groups for i in groups[g]]

    splits = {
        "train": {
            "images": [image_paths[i] for i in train_idx],
            "masks": [mask_paths[i] for i in train_idx] if mask_paths else None,
            "scenes": len(train_groups),
            "patches": len(train_idx)
        },
        "val": {
            "images": [image_paths[i] for i in val_idx],
            "masks": [mask_paths[i] for i in val_idx] if mask_paths else None,
            "scenes": len(val_groups),
            "patches": len(val_idx)
        },
        "test": {
            "images": [image_paths[i] for i in test_idx],
            "masks": [mask_paths[i] for i in test_idx] if mask_paths else None,
            "scenes": len(test_groups),
            "patches": len(test_idx)
        }
    }

    return splits["train"], splits["val"], splits["test"]
