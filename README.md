# Swin UNETR for BraTS 2020 Brain Tumor Segmentation

Brain tumor segmentation from multi-modal MRI using Swin UNETR with MC-Dropout uncertainty estimation and a compound loss combining Dice-Focal and distance-based boundary terms.

## Problem

Segment three overlapping tumor sub-regions from 4-channel MRI volumes:
- **WT** — Whole Tumor
- **TC** — Tumor Core
- **ET** — Enhancing Tumor

The dataset has severe class imbalance (background > 90% of voxels), which motivates the compound loss design.

## Method

- **Architecture**: Swin UNETR (Swin Transformer encoder + CNN decoder)
- **Loss**: DiceFocal + Distance Boundary Loss (weighted MSE near GT boundaries)
- **Uncertainty**: MC-Dropout with T=10 forward passes at inference
- **Training**: AdamW + cosine annealing with linear warmup, AMP, gradient clipping
- **Augmentation**: random flip, rotate, affine, Gaussian noise/smoothing, intensity jitter

## Dataset

BraTS 2020 Training Set — available on Kaggle:
[awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

Download and place under:
```
data/raw/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
```

Each patient folder should contain:
```
BraTS20_Training_001/
  BraTS20_Training_001_flair.nii
  BraTS20_Training_001_t1.nii
  BraTS20_Training_001_t1ce.nii
  BraTS20_Training_001_t2.nii
  BraTS20_Training_001_seg.nii
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start training (end-to-end: EDA → train → evaluate → visualize)
python train.py
```

All outputs (checkpoints, plots, metrics CSV) go to `results/`.

To change hyperparameters, edit `configs/config.py`.

## Results

| Region | Dice | IoU | Sensitivity | Specificity |
|--------|------|-----|-------------|-------------|
| WT     | ~0.86 | ~0.78 | ~0.86 | ~0.99 |
| TC     | ~0.90 | ~0.85 | ~0.92 | ~0.99 |
| ET     | ~0.74 | ~0.64 | ~0.77 | ~0.99 |

Results vary based on train/val split and number of epochs. The numbers above are approximate targets consistent with published Swin UNETR performance on BraTS 2020.

## Project Structure

```
.
├── configs/
│   └── config.py          # all hyperparameters and paths
├── src/
│   ├── data_utils.py      # case discovery, split, datalist
│   ├── transforms.py      # MONAI transform pipelines
│   ├── model.py           # Swin UNETR + MC-Dropout wrapper
│   ├── losses.py          # CompoundSegLoss (DiceFocal + Boundary)
│   ├── trainer.py         # train_one_epoch, validate, MetricTracker
│   ├── evaluate.py        # per-case metric computation
│   └── visualization.py   # all plotting functions
├── notebooks/             # Kaggle notebook
├── train.py               # main entry point
└── requirements.txt
```

## Notes

- Requires a GPU with at least 16GB VRAM for batch_size=1 with ROI (96,96,96)
- On a T4 (16GB), CACHE_RATE=0.25 and NUM_WORKERS=2 are stable

