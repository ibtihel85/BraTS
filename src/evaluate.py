from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from configs.config import cfg
from src.trainer import batch_to_image_label


def compute_full_metrics(model, loader, device, val_cases: List[Dict]) -> pd.DataFrame:
    model.eval()
    records = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        imgs, segs = batch_to_image_label(batch, device)
        with torch.no_grad(), autocast(device_type=device.type, enabled=cfg.AMP):
            logits = sliding_window_inference(
                imgs,
                roi_size=cfg.ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        preds_bin = (torch.sigmoid(logits) > 0.5).float()
        gt = segs[0].cpu().numpy()
        pr = preds_bin[0].cpu().numpy()

        row = {"case": val_cases[i]["pid"] if i < len(val_cases) else f"case_{i}"}
        for ci, region in enumerate(["WT", "TC", "ET"]):
            g = gt[ci].flatten()
            p = pr[ci].flatten()
            tp = (g * p).sum()
            fp = ((1 - g) * p).sum()
            fn = (g * (1 - p)).sum()
            tn = ((1 - g) * (1 - p)).sum()

            dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
            iou  = tp / (tp + fp + fn + 1e-6)
            sens = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)

            row[f"dice_{region}"] = float(dice)
            row[f"iou_{region}"]  = float(iou)
            row[f"sens_{region}"] = float(sens)
            row[f"spec_{region}"] = float(spec)

        records.append(row)

    return pd.DataFrame(records)


def print_summary(results_df: pd.DataFrame):
    metric_cols = [c for c in results_df.columns if c != "case"]
    summary = results_df[metric_cols].agg(["mean", "std", "median"])

    print("\n" + "=" * 70)
    print(" FINAL VALIDATION METRICS")
    print("=" * 70)
    for region in ["WT", "TC", "ET"]:
        d  = summary.loc["mean", f"dice_{region}"]
        ds = summary.loc["std",  f"dice_{region}"]
        io = summary.loc["mean", f"iou_{region}"]
        se = summary.loc["mean", f"sens_{region}"]
        sp = summary.loc["mean", f"spec_{region}"]
        print(f" {region:3s}  Dice: {d:.3f}±{ds:.3f} | IoU: {io:.3f} | Sens: {se:.3f} | Spec: {sp:.3f}")
    print("=" * 70)
    mean_dice = summary.loc["mean", ["dice_WT", "dice_TC", "dice_ET"]].mean()
    print(f" Mean Dice (WT/TC/ET): {mean_dice:.4f}")
    print("=" * 70)
