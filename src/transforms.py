import numpy as np
from monai.transforms import (
    Compose, LoadImaged, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd,
    NormalizeIntensityd, EnsureChannelFirstd, EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd, SpatialPadd,
)

from configs.config import cfg

KEYS_IMG = cfg.MODALITIES


def build_train_transforms():
    return Compose([
        LoadImaged(keys=KEYS_IMG + ["seg"]),
        EnsureChannelFirstd(keys=KEYS_IMG + ["seg"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
        Orientationd(keys=KEYS_IMG + ["seg"], axcodes="RAS"),
        Spacingd(keys=KEYS_IMG + ["seg"], pixdim=cfg.PIXDIM,
                 mode=("bilinear",) * 4 + ("nearest",)),
        NormalizeIntensityd(keys=KEYS_IMG, nonzero=True, channel_wise=True),
        CropForegroundd(keys=KEYS_IMG + ["seg"], source_key="t1ce"),
        SpatialPadd(keys=KEYS_IMG + ["seg"], spatial_size=cfg.ROI_SIZE, mode="constant"),
        RandCropByPosNegLabeld(keys=KEYS_IMG + ["seg"], label_key="seg",
                               spatial_size=cfg.ROI_SIZE, pos=3, neg=1, num_samples=2),
        RandFlipd(keys=KEYS_IMG + ["seg"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=KEYS_IMG + ["seg"], prob=0.5, max_k=3),
        RandAffined(keys=KEYS_IMG + ["seg"], prob=0.5, rotate_range=(np.pi / 12,) * 3,
                    scale_range=(0.1,) * 3, translate_range=(5,) * 3, padding_mode="zeros"),
        RandGaussianNoised(keys=KEYS_IMG, prob=0.2, std=0.02),
        RandGaussianSmoothd(keys=KEYS_IMG, prob=0.2),
        RandScaleIntensityd(keys=KEYS_IMG, prob=0.3, factors=0.1),
        RandShiftIntensityd(keys=KEYS_IMG, prob=0.3, offsets=0.1),
        EnsureTyped(keys=KEYS_IMG + ["seg"]),
    ])


def build_val_transforms():
    return Compose([
        LoadImaged(keys=KEYS_IMG + ["seg"]),
        EnsureChannelFirstd(keys=KEYS_IMG + ["seg"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
        Orientationd(keys=KEYS_IMG + ["seg"], axcodes="RAS"),
        Spacingd(keys=KEYS_IMG + ["seg"], pixdim=cfg.PIXDIM,
                 mode=("bilinear",) * 4 + ("nearest",)),
        NormalizeIntensityd(keys=KEYS_IMG, nonzero=True, channel_wise=True),
        CropForegroundd(keys=KEYS_IMG + ["seg"], source_key="t1ce"),
        SpatialPadd(keys=KEYS_IMG + ["seg"], spatial_size=cfg.ROI_SIZE, mode="constant"),
        EnsureTyped(keys=KEYS_IMG + ["seg"]),
    ])
