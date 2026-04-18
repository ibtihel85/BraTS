from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm

from configs.config import cfg


dice_metric = DiceMetric(
    include_background=True, reduction="mean_batch", get_not_nans=False
)
hausdorff_metric = HausdorffDistanceMetric(
    include_background=True, percentile=95, reduction="mean_batch"
)


def batch_to_image_label(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.cat([batch[mod] for mod in cfg.MODALITIES], dim=1).to(device)
    segs = batch["seg"].to(device)
    return imgs, segs


class MetricTracker:
    def __init__(self):
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_dice_wt": [], "val_dice_tc": [], "val_dice_et": [],
            "val_hd95_wt": [], "val_hd95_tc": [], "val_hd95_et": [],
            "lr": [],
        }

    def update(self, key: str, val: float):
        self.history[key].append(val)

    def latest(self, key: str) -> float:
        return self.history[key][-1] if self.history[key] else 0.0


def train_one_epoch(model, loader, optimizer, criterion, scaler: GradScaler, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  [Train]", leave=False)
    for batch in pbar:
        imgs, segs = batch_to_image_label(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=cfg.AMP):
            preds = model(imgs)
            loss, breakdown = criterion(preds, segs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += breakdown["total"]
        n_batches += 1
        pbar.set_postfix({"loss": f"{breakdown['total']:.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict:
    model.eval()
    dice_metric.reset()
    hausdorff_metric.reset()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  [Val]  ", leave=False)
    for batch in pbar:
        imgs, segs = batch_to_image_label(batch, device)

        with autocast(device_type=device.type, enabled=cfg.AMP):
            preds = sliding_window_inference(
                inputs=imgs,
                roi_size=cfg.ROI_SIZE,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
            loss, breakdown = criterion(preds, segs)

        preds_bin = (torch.sigmoid(preds) > 0.5).float()
        dice_metric(y_pred=preds_bin, y=segs)
        try:
            hausdorff_metric(y_pred=preds_bin, y=segs)
        except Exception:
            pass

        total_loss += breakdown["total"]
        n_batches += 1

    dice_vals = dice_metric.aggregate().cpu().numpy()
    try:
        hd_vals = hausdorff_metric.aggregate().cpu().numpy()
    except Exception:
        hd_vals = np.array([0.0, 0.0, 0.0])

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice_wt": float(dice_vals[0]),
        "dice_tc": float(dice_vals[1]),
        "dice_et": float(dice_vals[2]),
        "dice_mean": float(dice_vals.mean()),
        "hd95_wt": float(hd_vals[0]),
        "hd95_tc": float(hd_vals[1]),
        "hd95_et": float(hd_vals[2]),
    }
