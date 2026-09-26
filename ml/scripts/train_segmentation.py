# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — Honest Sentinel-1 U-Net Segmentation Training Pipeline
Implements:
1. Spatio-temporal group cross-validation (zero scene overlap between train/val/test).
2. Composite loss: Focal Loss + Tversky Loss (beta=0.7) + Boundary Laplacian Loss.
3. Strict evaluation: Mean IoU, Dice F1, Boundary IoU, Precision, Recall.
4. Auto-export to ONNX and verifiable benchmark_report.json generation.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import SARDataset, create_spatiotemporal_splits
from ml.models.unet import CompactSARUNet, UNetS1SAR
from ml.models.losses import CompositeSARLoss
from ml.evaluation.metrics import evaluate_predictions, save_benchmark_report
from ml.scripts.export_onnx import export_segmentation_onnx

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_sz = images.size(0)
        total_loss += loss.item() * batch_sz
        total_samples += batch_sz

    return total_loss / max(total_samples, 1)


def evaluate_model(model, loader, device, threshold=0.40):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits) if not isinstance(model, CompactSARUNet) else logits
            
            probs_np = probs.squeeze(1).cpu().numpy()
            targets_np = targets.squeeze(1).cpu().numpy()

            for i in range(probs_np.shape[0]):
                all_preds.append(probs_np[i])
                all_targets.append(targets_np[i])

    return evaluate_predictions(all_preds, all_targets, threshold=threshold)


def run_pipeline(args):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run the training pipeline.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Spill Sense — SAR Segmentation Model Training Pipeline")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print(f"=======================================================\n")

    data_dir = Path(args.data_dir)
    images = sorted(list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png")))
    
    if not images:
        print(f"No image files found in {data_dir}. Falling back to demo mode export...")
        export_segmentation_onnx(Path(args.output_onnx), img_size=args.img_size)
        return

    print(f"Discovered {len(images)} SAR image patches.")
    masks_dir = data_dir / "masks"
    masks = [masks_dir / img.name for img in images] if masks_dir.exists() else None

    # Spatio-Temporal Group Splitting
    print("Executing Spatio-Temporal Group Splitting (preventing satellite swath leakage)...")
    train_split, val_split, test_split = create_spatiotemporal_splits(
        images, masks, train_ratio=0.70, val_ratio=0.15, seed=42
    )

    print(f"Split results:")
    print(f"  Train: {train_split['patches']} patches across {train_split['scenes']} satellite acquisitions")
    print(f"  Val:   {val_split['patches']} patches across {val_split['scenes']} satellite acquisitions")
    print(f"  Test:  {test_split['patches']} patches across {test_split['scenes']} satellite acquisitions\n")

    train_ds = SARDataset(train_split["images"], train_split["masks"], img_size=args.img_size, augment=True)
    val_ds = SARDataset(val_split["images"], val_split["masks"], img_size=args.img_size, augment=False)
    test_ds = SARDataset(test_split["images"], test_split["masks"], img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CompactSARUNet(in_channels=1, out_channels=1).to(device)
    criterion = CompositeSARLoss(w_focal=0.4, w_tversky=0.4, w_boundary=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_iou = 0.0
    best_weights_path = Path(args.output_dir) / "best_segmenter.pt"
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'Ep':>3} | {'TrLoss':>7} | {'ValIoU':>7} | {'ValDice':>7} | {'ValB-IoU':>8} | {'Prec':>6} | {'Rec':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, threshold=0.40)

        v_iou = val_metrics["mean_iou"]
        v_dice = val_metrics["mean_dice_f1"]
        v_biou = val_metrics["mean_boundary_iou"]
        v_prec = val_metrics["precision"]
        v_rec = val_metrics["recall"]

        scheduler.step(v_iou)

        print(f"{epoch:3d} | {tr_loss:7.4f} | {v_iou*100:6.1f}% | {v_dice*100:6.1f}% | {v_biou*100:7.1f}% | {v_prec*100:5.1f}% | {v_rec*100:5.1f}%")

        if v_iou > best_val_iou:
            best_val_iou = v_iou
            torch.save(model.state_dict(), str(best_weights_path))

    # Final Test Set Evaluation
    print("\n" + "=" * 55)
    print("FINAL TEST SET EVALUATION (Unseen Geographic Passes)")
    print("=" * 55)
    if best_weights_path.exists():
        model.load_state_dict(torch.load(str(best_weights_path), map_location=device))

    test_metrics = evaluate_model(model, test_loader, device, threshold=0.40)
    print(f"Mean IoU:         {test_metrics['mean_iou']*100:.2f}%")
    print(f"Dice F1-Score:    {test_metrics['mean_dice_f1']:.4f}")
    print(f"Boundary IoU:     {test_metrics['mean_boundary_iou']*100:.2f}%")
    print(f"Precision:        {test_metrics['precision']:.4f}")
    print(f"Recall:           {test_metrics['recall']:.4f}")

    # Export ONNX and save verifiable benchmark report
    export_segmentation_onnx(Path(args.output_onnx), img_size=args.img_size, weights_path=best_weights_path)
    report_path = Path(PROJECT_ROOT) / "backend" / "app" / "data" / "benchmark_report.json"
    save_benchmark_report(test_metrics, report_path)
    print(f"\nPipeline successfully completed! Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Sentinel-1 SAR Segmentation Model")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "frontend" / "public" / "demo-sar"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "ml" / "checkpoints"))
    parser.add_argument("--output_onnx", type=str, default=str(PROJECT_ROOT / "frontend" / "public" / "models" / "oil_segmenter.onnx"))
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
