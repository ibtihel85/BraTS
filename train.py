import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader as MonaiLoader
from monai.utils import set_determinism
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from configs.config import cfg
from src.data_utils import discover_cases, split_cases, make_datalist, analyze_label_distribution
from src.transforms import build_train_transforms, build_val_transforms
from src.model import SwinUNETRWithUncertainty
from src.losses import CompoundSegLoss
from src.trainer import train_one_epoch, validate, MetricTracker
from src.evaluate import compute_full_metrics, print_summary
from src.visualization import (
    plot_case_sample,
    plot_label_distribution,
    plot_training_curves,
    visualize_predictions,
    visualize_uncertainty,
    visualize_attention,
    plot_final_metrics,
    plot_sota_comparison,
)


def lr_lambda(epoch: int) -> float:
    if epoch < cfg.WARMUP_EPOCHS:
        return float(epoch + 1) / float(cfg.WARMUP_EPOCHS)
    progress = (epoch - cfg.WARMUP_EPOCHS) / (cfg.EPOCHS - cfg.WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def main():
    set_determinism(seed=cfg.SEED)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────
    print(f"\nLoading cases from: {cfg.TRAIN_DIR}")
    all_cases = discover_cases(cfg.TRAIN_DIR)
    print(f"Total valid cases: {len(all_cases)}")

    train_cases, val_cases = split_cases(all_cases)
    print(f"Train: {len(train_cases)} | Val: {len(val_cases)}")

    # EDA
    if all_cases:
        plot_case_sample(all_cases[0], cfg.OUTPUT_DIR)
        label_counts = analyze_label_distribution(all_cases, n_samples=20)
        plot_label_distribution(label_counts, cfg.OUTPUT_DIR)

    # ── Datasets & Loaders ────────────────────────────────
    train_ds = CacheDataset(make_datalist(train_cases),
                            transform=build_train_transforms(), cache_rate=cfg.CACHE_RATE)
    val_ds   = CacheDataset(make_datalist(val_cases),
                            transform=build_val_transforms(), cache_rate=cfg.CACHE_RATE)

    train_loader = MonaiLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = MonaiLoader(val_ds, batch_size=1, shuffle=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── Model, Loss, Optimizer ────────────────────────────
    model     = SwinUNETRWithUncertainty().to(device)
    criterion = CompoundSegLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=cfg.AMP)
    tracker   = MetricTracker()

    # ── Training Loop ─────────────────────────────────────
    best_dice  = -1.0
    start_time = time.time()

    print("\n" + "=" * 65)
    print(f" Starting Training: Swin UNETR on BraTS 2020")
    print(f" Epochs: {cfg.EPOCHS}  |  Batch: {cfg.BATCH_SIZE}  |  AMP: {cfg.AMP}")
    print("=" * 65)

    for epoch in range(1, cfg.EPOCHS + 1):
        ep_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        tracker.update("train_loss", train_loss)
        tracker.update("lr", current_lr)

        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.EPOCHS:
            val_results = validate(model, val_loader, criterion, device)

            tracker.update("val_loss",    val_results["loss"])
            tracker.update("val_dice_wt", val_results["dice_wt"])
            tracker.update("val_dice_tc", val_results["dice_tc"])
            tracker.update("val_dice_et", val_results["dice_et"])
            tracker.update("val_hd95_wt", val_results["hd95_wt"])
            tracker.update("val_hd95_tc", val_results["hd95_tc"])
            tracker.update("val_hd95_et", val_results["hd95_et"])

            mean_dice = val_results["dice_mean"]
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "best_dice":  best_dice,
                }, cfg.CHECKPOINT)
                ckpt_str = " -> SAVED"
            else:
                ckpt_str = ""

            ep_time = time.time() - ep_start
            print(
                f"Ep {epoch:>3}/{cfg.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Dice WT/TC/ET: "
                f"{val_results['dice_wt']:.3f}/{val_results['dice_tc']:.3f}/{val_results['dice_et']:.3f} | "
                f"Mean: {mean_dice:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"{ep_time:.0f}s{ckpt_str}"
            )
        else:
            ep_time = time.time() - ep_start
            print(
                f"Ep {epoch:>3}/{cfg.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{ep_time:.0f}s"
            )

    total_time = (time.time() - start_time) / 60
    print("=" * 65)
    print(f" Training complete in {total_time:.1f} min")
    print(f" Best Mean Dice: {best_dice:.4f}")
    print("=" * 65)

    plot_training_curves(tracker, cfg.OUTPUT_DIR)

    # ── Load Best & Evaluate ──────────────────────────────
    if cfg.CHECKPOINT.exists():
        ckpt = torch.load(cfg.CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\nLoaded best model (epoch {ckpt['epoch']}, dice={ckpt['best_dice']:.4f})")

    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        pid = val_cases[i]["pid"] if i < len(val_cases) else f"case_{i}"
        visualize_predictions(model, batch, device, case_id=pid, output_dir=cfg.OUTPUT_DIR)

    val_batch = next(iter(val_loader))
    pid = val_cases[0]["pid"] if val_cases else "case_0"
    visualize_uncertainty(model, val_batch, device, case_id=pid, output_dir=cfg.OUTPUT_DIR)
    visualize_attention(model, val_batch, device, case_id=pid, output_dir=cfg.OUTPUT_DIR)

    results_df = compute_full_metrics(model, val_loader, device, val_cases)
    results_df.to_csv(cfg.OUTPUT_DIR / "val_metrics.csv", index=False)
    print_summary(results_df)

    plot_final_metrics(results_df, cfg.OUTPUT_DIR)
    plot_sota_comparison(results_df, cfg.OUTPUT_DIR)

    print(f"\nAll outputs saved to: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
