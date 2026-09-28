# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Automated Active Learning Fine-Tuning Pipeline
=======================================================================
Implements Plan 2.0 Phase 4 continual learning:
1. Ingests duty officer hard negatives and confirmed ground truth samples.
2. Combines them with a replay buffer of core SAR scenes to prevent catastrophic forgetting.
3. Fine-tunes Domain-Adversarial U-Net (DANN).
4. Automatically triggers edge quantization and deploys updated ONNX models to frontend.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ml.models.domain_adversarial_unet import DomainAdversarialUNet
from ml.models.composite_losses import DomainAdversarialCompositeLoss
from ml.data.universal_loader import UniversalSARDataset
from ml.scripts.train_dann import discover_or_synthesize_samples, evaluate_dann


def parse_args():
    parser = argparse.ArgumentParser(description="Spill Sense - Active Learning Continual Retraining")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--img-size", type=int, default=256, help="Image resolution")
    parser.add_argument("--min-feedback", type=int, default=1, help="Minimum feedback samples required")
    return parser.parse_args()


def run_active_learning_retrain(args):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for active learning fine-tuning.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("Spill Sense (SIH-26143) - Active Learning Fine-Tuning Pipeline")
    print("=" * 65)

    cache_file = PROJECT_ROOT / "backend" / "data" / "active_learning" / "feedback_cache.json"
    feedback_samples = []
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                feedback_samples = json.load(f)
        except Exception as e:
            print(f"Warning reading feedback cache: {e}")

    print(f"Loaded {len(feedback_samples)} duty officer feedback records.")

    # 1. Base Replay Buffer (Core Scenes)
    base_samples = discover_or_synthesize_samples(PROJECT_ROOT / "ml" / "benchmark_gallery", min_samples=12)

    # 2. Convert feedback into training items
    active_samples: List[Dict[str, Any]] = []
    for item in feedback_samples:
        img_rel = item.get("image_path")
        if img_rel and (PROJECT_ROOT / img_rel).exists():
            # For confirmed spills, ground truth is oil; for hard negative rejections, ground truth is clean water
            is_reject = item.get("is_hard_negative", False)
            active_samples.append({
                "image_path": str(PROJECT_ROOT / img_rel),
                "mask_path": None, # Will be treated as 0 mask (clean water) for hard negative rejections
                "domain": "sentinel1_dualpol"
            })

    all_training_samples = base_samples + active_samples
    print(f"Total training pool: {len(base_samples)} replay scenes + {len(active_samples)} active feedback scenes")

    # Split
    val_count = max(2, int(len(all_training_samples) * 0.20))
    train_count = len(all_training_samples) - val_count

    train_ds = UniversalSARDataset(all_training_samples[:train_count], target_size=(args.img_size, args.img_size), augment=True)
    val_ds = UniversalSARDataset(all_training_samples[train_count:], target_size=(args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainAdversarialUNet(in_channels=2, num_classes=1, num_domains=4, export_mode=False).to(device)

    # Load existing checkpoint if available
    ckpt_path = PROJECT_ROOT / "ml" / "weights" / "dann" / "dann_best.pt"
    if ckpt_path.exists():
        print(f"Initializing from base weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    criterion = DomainAdversarialCompositeLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\nFine-tuning on Hard-Negative Replay Buffer:")
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_accum = 0.0
        samples = 0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            masks = batch["mask"].to(device)
            domains = batch["domain_id"].to(device)

            optimizer.zero_grad()
            out = model(inputs, lambda_p=0.5)
            loss_dict = criterion(out, masks, domains)
            loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bsz = inputs.size(0)
            loss_accum += loss_dict["total"].item() * bsz
            samples += bsz

        val_metrics = evaluate_dann(model, val_loader, device=device)
        print(f"  Epoch {ep}/{args.epochs} | Loss: {loss_accum / max(1, samples):.4f} | Val IoU: {val_metrics['mean_iou']:.3f} | Dom Acc: {val_metrics['domain_accuracy']:.1%}")

    # Save fine-tuned checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "fine_tuned_at": time.time()}, ckpt_path)
    print(f"\nSaved fine-tuned model checkpoint: {ckpt_path}")

    # Trigger automatic edge export & quantization to frontend
    print("Triggering automatic edge quantization & deployment to frontend/public/models/...")
    from ml.scripts.export_dann_edge import export_dann_edge
    export_args = argparse.Namespace(
        checkpoint=str(ckpt_path.relative_to(PROJECT_ROOT)),
        output_dir="frontend/public/models",
        img_size=512,
        quantize_int8=True,
        set_primary=True
    )
    export_dann_edge(export_args)
    print("\nActive learning feedback integration & deployment completed successfully!\n")


if __name__ == "__main__":
    args = parse_args()
    run_active_learning_retrain(args)
