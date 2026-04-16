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

    def predict_with_uncertainty(self, x, n_passes: int = cfg.MC_PASSES):
        self.train()
        preds = []
        try:
            with torch.no_grad():
                for _ in range(n_passes):
                    logits = sliding_window_inference(
                        x, roi_size=cfg.ROI_SIZE, sw_batch_size=1,
                        predictor=self.model, overlap=0.5
                    )
                    preds.append(self.sigmoid(logits))
        finally:
            self.eval()
        preds = torch.stack(preds)
        return preds.mean(0), preds.var(0)
