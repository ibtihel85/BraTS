import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

from configs.config import cfg


class SwinUNETRWithUncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SwinUNETR(
            spatial_dims=3,
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            feature_size=cfg.FEATURE_SIZE,
            use_checkpoint=cfg.USE_CHECKPOINT,
            dropout_path_rate=cfg.DROPOUT,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)

    
