from typing import Dict, Tuple

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss


class CompoundSegLoss(nn.Module):
    def __init__(self, w_dice_focal: float = 1.0):
        super().__init__()
        self.dice_focal = DiceFocalLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            smooth_nr=0,
            smooth_dr=1e-5,
            batch=True,
            gamma=2.0,
            weight=(0.5, 1.0, 2.0),
        )
        self.w_dice_focal = w_dice_focal

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        l_df = self.dice_focal(pred, target)
        total = self.w_dice_focal * l_df

        breakdown = {
            "dice_focal": l_df.item(),
            "total": total.item(),
        }

        return total, breakdown