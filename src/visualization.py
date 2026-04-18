from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.amp import autocast
from monai.inferers import sliding_window_inference

from configs.config import cfg
from src.trainer import batch_to_image_label


def plot_case_sample(case: Dict, output_dir, slice_idx: int = 100):
    import nibabel as nib

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.patch.set_facecolor("#0d0d0d")
    cmap_seg = ListedColormap(["#000000", "#4CAF50", "#FF9800", "#F44336"])

    for i, mod in enumerate(cfg.MODALITIES):
        img = nib.load(case[mod]).get_fdata()[:, :, slice_idx]
        axes[i].imshow(img.T, cmap="gray", origin="lower")
        axes[i].set_title(f"{mod.upper()}", color="white", fontsize=12, fontweight="bold")
        axes[i].axis("off")

    seg = nib.load(case["seg"]).get_fdata()[:, :, slice_idx]
    seg_display = np.where(seg == 4, 3, seg)
    axes[4].imshow(seg.T, cmap="gray", origin="lower", alpha=0.6)
    axes[4].imshow(seg_display.T, cmap=cmap_seg, origin="lower", alpha=0.6, vmin=0, vmax=3)
    axes[4].set_title("Segmentation", color="white", fontsize=12, fontweight="bold")
    axes[4].axis("off")

    t1ce = nib.load(case["t1ce"]).get_fdata()[:, :, slice_idx]
    axes[5].imshow(t1ce.T, cmap="gray", origin="lower")
    axes[5].imshow(seg_display.T, cmap=cmap_seg, origin="lower", alpha=0.45, vmin=0, vmax=3)
    axes[5].set_title("T1ce + Mask", color="white", fontsize=12, fontweight="bold")
    axes[5].axis("off")

    patches = [
        mpatches.Patch(color="#4CAF50", label="Edema (WT)"),
        mpatches.Patch(color="#FF9800", label="Core (TC)"),
        mpatches.Patch(color="#F44336", label="Enhancing (ET)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#1a1a1a", labelcolor="white", fontsize=10, framealpha=0.9)
    plt.suptitle(f"Patient: {case['pid']}", color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_case.png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def plot_label_distribution(label_counts: Dict, output_dir):
    total = sum(label_counts.values())
    labels_names = ["Background", "NCR/NET (1)", "Edema (2)", "ET (4)"]
    values = [label_counts[k] for k in [0, 1, 2, 4]]
    percents = [v / total * 100 for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1a1a")

    colors = ["#607D8B", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax1.bar(labels_names, percents, color=colors, edgecolor="#333")
    ax1.set_ylabel("% of total voxels", color="white")
    ax1.set_title("Class Imbalance (Voxel Level)", color="white", fontweight="bold")
    ax1.tick_params(colors="white")
    for bar, pct in zip(bars, percents):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{pct:.1f}%", ha="center", color="white", fontsize=10)

    ax2.pie(values[1:], labels=labels_names[1:], colors=colors[1:],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white"})
    ax2.set_title("Tumor Region Breakdown (excl. background)", color="white", fontweight="bold")

    plt.suptitle("Severe Class Imbalance — Justifies Dice + Focal Loss",
                 color="#FFB74D", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#111111")
    plt.close()


def plot_training_curves(tracker, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        ax.grid(True, alpha=0.2, color="#444")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    val_epochs = list(range(cfg.VAL_FREQ,
                            len(tracker.history["val_loss"]) * cfg.VAL_FREQ + 1,
                            cfg.VAL_FREQ))

    ax = axes[0, 0]
    ax.plot(tracker.history["train_loss"], color="#42A5F5", lw=2, label="Train")
    if tracker.history["val_loss"]:
        ax.plot(val_epochs[:len(tracker.history["val_loss"])],
                tracker.history["val_loss"], color="#EF5350", lw=2, label="Val")
    ax.set_title("Loss", color="white", fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")

    ax = axes[0, 1]
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    ve = val_epochs[:len(tracker.history["val_dice_wt"])]
    for key, color, name in zip(
        ["val_dice_wt", "val_dice_tc", "val_dice_et"], colors, ["WT", "TC", "ET"]
    ):
        if tracker.history[key]:
            ax.plot(ve, tracker.history[key], color=color, lw=2, label=name, marker="o", ms=4)
    ax.set_ylim(0, 1)
    ax.set_title("Dice per Class", color="white", fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")
    ax.axhline(0.9, color="white", ls="--", alpha=0.3, lw=1)

    ax = axes[1, 0]
    ve2 = val_epochs[:len(tracker.history["val_hd95_wt"])]
    for key, color, name in zip(
        ["val_hd95_wt", "val_hd95_tc", "val_hd95_et"], colors, ["WT", "TC", "ET"]
    ):
        if tracker.history[key]:
            ax.plot(ve2, tracker.history[key], color=color, lw=2, label=name, marker="s", ms=4)
    ax.set_title("HD95 (mm) — lower is better", color="white", fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")

    ax = axes[1, 1]
    ax.plot(tracker.history["lr"], color="#CE93D8", lw=2)
    ax.set_title("Learning Rate Schedule", color="white", fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.set_yscale("log")

    plt.suptitle("Training History — Swin UNETR (BraTS 2020)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def visualize_predictions(model, batch: Dict, device, case_id: str, output_dir):
    model.eval()
    imgs, segs = batch_to_image_label(batch, device)

    with torch.no_grad(), autocast(device_type=device.type, enabled=cfg.AMP):
        logits = sliding_window_inference(
            imgs, roi_size=cfg.ROI_SIZE, sw_batch_size=1, predictor=model
        )
    preds_bin = (torch.sigmoid(logits) > 0.5).float()

    img_np  = imgs[0, 0].cpu().numpy()
    seg_np  = segs[0].cpu().numpy()
    pred_np = preds_bin[0].cpu().numpy()

    tumor_per_slice = seg_np.sum(axis=(0, 1, 2))
    best_slice = int(np.argmax(tumor_per_slice)) if tumor_per_slice.sum() > 0 else img_np.shape[2] // 2

    regions = [("WT", 0, "#4CAF50"), ("TC", 1, "#FF9800"), ("ET", 2, "#F44336")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor("#0d0d0d")
    bg_img = img_np[:, :, best_slice].T

    for row_i, (mask_np, row_title) in enumerate([(seg_np, "Ground Truth"), (pred_np, "Prediction")]):
        axes[row_i, 0].imshow(bg_img, cmap="gray", origin="lower")
        axes[row_i, 0].set_title(f"FLAIR ({row_title})", color="white", fontsize=10)
        axes[row_i, 0].axis("off")

        for col_i, (region_name, ch, color) in enumerate(regions):
            region_mask = mask_np[ch, :, :, best_slice].T
            axes[row_i, col_i + 1].imshow(bg_img, cmap="gray", origin="lower")
            masked = np.ma.masked_where(region_mask == 0, region_mask)
            axes[row_i, col_i + 1].imshow(masked, cmap=ListedColormap([color]),
                                           origin="lower", alpha=0.6, vmin=0, vmax=1)
            if row_i == 1:
                gt_c = seg_np[ch, :, :, best_slice]
                pr_c = mask_np[ch, :, :, best_slice]
                intersection = (gt_c * pr_c).sum()
                dice_val = 2 * intersection / (gt_c.sum() + pr_c.sum() + 1e-6)
                axes[row_i, col_i + 1].set_title(
                    f"{region_name} (Dice={dice_val:.3f})", color=color, fontsize=10, fontweight="bold"
                )
            else:
                axes[row_i, col_i + 1].set_title(f"{region_name}", color=color, fontsize=10, fontweight="bold")
            axes[row_i, col_i + 1].axis("off")

    for ax in axes.flat:
        ax.set_facecolor("#0d0d0d")

    plt.suptitle(f"Segmentation: {case_id} | Axial Slice {best_slice}",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f"prediction_{case_id}.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def visualize_uncertainty(model, batch: Dict, device, case_id: str, output_dir):
    imgs, segs = batch_to_image_label(batch, device)
    mean_pred, uncertainty = model.predict_with_uncertainty(imgs, n_passes=cfg.MC_PASSES)

    img_np  = imgs[0, 0].cpu().numpy()
    mean_np = mean_pred[0].cpu().numpy()
    unc_np  = uncertainty[0].cpu().numpy()
    seg_np  = segs[0].cpu().numpy()

    best_slice = img_np.shape[2] // 2
    bg = img_np[:, :, best_slice].T

    region_names = ["WT", "TC", "ET"]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor("#0d0d0d")

    for r, (name, color) in enumerate(zip(region_names, colors)):
        axes[r, 0].imshow(bg, cmap="gray", origin="lower")
        gt_mask = np.ma.masked_where(seg_np[r, :, :, best_slice].T == 0,
                                     seg_np[r, :, :, best_slice].T)
        axes[r, 0].imshow(gt_mask, cmap=ListedColormap([color]), origin="lower", alpha=0.6)
        axes[r, 0].set_title(f"{name} — GT", color=color, fontweight="bold")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(bg, cmap="gray", origin="lower")
        pred_slice = mean_np[r, :, :, best_slice].T
        axes[r, 1].imshow(pred_slice, cmap="hot", origin="lower", alpha=0.6, vmin=0, vmax=1)
        axes[r, 1].set_title(f"{name} — Mean Pred", color="white", fontweight="bold")
        axes[r, 1].axis("off")

        bin_pred = (pred_slice > 0.5).astype(float)
        axes[r, 2].imshow(bg, cmap="gray", origin="lower")
        bin_masked = np.ma.masked_where(bin_pred == 0, bin_pred)
        axes[r, 2].imshow(bin_masked, cmap=ListedColormap([color]), origin="lower", alpha=0.6)
        axes[r, 2].set_title(f"{name} — Binary Pred", color=color, fontweight="bold")
        axes[r, 2].axis("off")

        unc_slice = unc_np[r, :, :, best_slice].T
        im = axes[r, 3].imshow(unc_slice, cmap="plasma", origin="lower")
        axes[r, 3].set_title(f"{name} — Uncertainty (σ²)", color="#CE93D8", fontweight="bold")
        axes[r, 3].axis("off")
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_facecolor("#111111")

    plt.suptitle(f"MC-Dropout Uncertainty Maps (T={cfg.MC_PASSES} passes) | {case_id}",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f"uncertainty_{case_id}.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"   Global uncertainty stats — Mean: {unc_np.mean():.4f}, Max: {unc_np.max():.4f}")


def visualize_attention(model, batch: Dict, device, case_id: str, output_dir):
    class AttentionExtractor:
        def __init__(self, m: nn.Module):
            self.attention_maps = []
            self.hooks = []
            for name, module in m.named_modules():
                if "attn" in name.lower() and hasattr(module, "relative_position_bias_table"):
                    self.hooks.append(module.register_forward_hook(self._hook_fn))

        def _hook_fn(self, module, input, output):
            if isinstance(output, tuple):
                self.attention_maps.append(output[0].detach().cpu())
            else:
                self.attention_maps.append(output.detach().cpu())

        def clear(self):
            self.attention_maps.clear()

        def remove_hooks(self):
            for h in self.hooks:
                h.remove()
            self.hooks.clear()

    extractor = AttentionExtractor(model.model)
    imgs, _ = batch_to_image_label(batch, device)

    model.eval()
    extractor.clear()
    with torch.no_grad():
        _ = sliding_window_inference(
            imgs, roi_size=cfg.ROI_SIZE, sw_batch_size=1, predictor=model, overlap=0.5
        )
    extractor.remove_hooks()

    img_np = imgs[0, 0].cpu().numpy()
    mid_slice = img_np.shape[2] // 2

    if not extractor.attention_maps:
        imgs_req = imgs.clone().requires_grad_(True)
        model.eval()
        logits = model(imgs_req)
        logits[:, 2].sum().backward()
        saliency = imgs_req.grad[0, 0].abs().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0d0d0d")
        bg = img_np[:, :, mid_slice].T

        axes[0].imshow(bg, cmap="gray", origin="lower")
        axes[0].set_title("FLAIR Input", color="white")
        axes[0].axis("off")

        axes[1].imshow(saliency[:, :, mid_slice].T, cmap="hot", origin="lower")
        axes[1].set_title("Gradient Saliency (ET)", color="#F44336", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(bg, cmap="gray", origin="lower", alpha=0.7)
        axes[2].imshow(saliency[:, :, mid_slice].T, cmap="jet", origin="lower", alpha=0.5)
        axes[2].set_title("Overlay", color="white")
        axes[2].axis("off")

        for ax in axes:
            ax.set_facecolor("#0d0d0d")

        plt.suptitle(f"Gradient Saliency Map | {case_id}",
                     color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"saliency_{case_id}.png", dpi=150,
                    bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        return

    last_attn = extractor.attention_maps[-1].float()
    attn_avg = last_attn.mean(dim=(0, 1)).unsqueeze(0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")
    sns.heatmap(attn_avg.numpy(), ax=ax, cmap="magma", cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Swin Attention Heatmap (Last Stage) | {case_id}",
                 color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"attention_{case_id}.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def plot_final_metrics(results_df, output_dir):
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.grid(True, alpha=0.2, color="#444")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    region_colors = {"WT": "#4CAF50", "TC": "#FF9800", "ET": "#F44336"}

    dice_data = {r: results_df[f"dice_{r}"].values for r in ["WT", "TC", "ET"]}
    parts = axes[0].violinplot(list(dice_data.values()), positions=[1, 2, 3], showmedians=True)
    for pc, color in zip(parts["bodies"], ["#4CAF50", "#FF9800", "#F44336"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(["WT", "TC", "ET"], color="white")
    axes[0].set_ylabel("Dice Score", color="white")
    axes[0].set_title("Dice Distribution per Region", color="white", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.9, color="white", ls="--", alpha=0.3)

    iou_means = [results_df[f"iou_{r}"].mean() for r in ["WT", "TC", "ET"]]
    iou_stds  = [results_df[f"iou_{r}"].std()  for r in ["WT", "TC", "ET"]]
    bars = axes[1].bar(["WT", "TC", "ET"], iou_means,
                       color=["#4CAF50", "#FF9800", "#F44336"],
                       yerr=iou_stds, capsize=5, edgecolor="#333")
    axes[1].set_ylabel("IoU (Jaccard)", color="white")
    axes[1].set_title("Mean IoU per Region", color="white", fontweight="bold")
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars, iou_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", color="white", fontsize=11)

    for r, color in region_colors.items():
        axes[2].scatter(results_df[f"spec_{r}"], results_df[f"sens_{r}"],
                        label=r, color=color, alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
    axes[2].set_xlabel("Specificity", color="white")
    axes[2].set_ylabel("Sensitivity", color="white")
    axes[2].set_title("Sensitivity vs Specificity", color="white", fontweight="bold")
    axes[2].legend(facecolor="#2a2a2a", labelcolor="white")

    plt.suptitle("Swin UNETR — BraTS 2020 Validation Metrics",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "final_metrics.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def plot_sota_comparison(results_df, output_dir):
    sota_data = {
        "Method": [
            "3D U-Net (baseline)",
            "nnU-Net (Isensee 2021)",
            "TransBTS (Wang 2021)",
            "Swin UNETR (Tang 2022)",
            "MedNeXt (Roy 2023)",
            "nnFormer (Zhou 2023)",
            "This Work",
        ],
        "ET Dice": [0.736, 0.823, 0.810, 0.832, 0.844, 0.838, None],
        "TC Dice": [0.782, 0.881, 0.862, 0.893, 0.896, 0.888, None],
        "WT Dice": [0.891, 0.918, 0.919, 0.926, 0.932, 0.921, None],
        "Year":    [2016, 2021, 2021, 2022, 2023, 2023, 2025],
    }

    if not results_df.empty:
        sota_data["ET Dice"][-1] = round(results_df["dice_ET"].mean(), 3)
        sota_data["TC Dice"][-1] = round(results_df["dice_TC"].mean(), 3)
        sota_data["WT Dice"][-1] = round(results_df["dice_WT"].mean(), 3)
    else:
        sota_data["ET Dice"][-1] = "—"
        sota_data["TC Dice"][-1] = "—"
        sota_data["WT Dice"][-1] = "—"

    import pandas as pd
    sota_df = pd.DataFrame(sota_data)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.axis("off")

    col_labels = sota_df.columns.tolist()
    cell_text  = sota_df.values.tolist()
    colors_row = [["#1a2a3a"] * len(col_labels)] * (len(sota_df) - 1) + [["#1a3a2a"] * len(col_labels)]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center", cellColours=colors_row)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333")
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="#80CBC4", fontweight="bold")

    ax.set_title("SOTA Comparison — BraTS 2020 Benchmark",
                 color="white", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "sota_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
