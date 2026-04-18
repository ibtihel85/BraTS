from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceFocalLoss


class DistanceBoundaryLoss(nn.Module):
    def __init__(self, max_distance: float = 5.0, sigma: float = 2.0):
        super().__init__()
        self.max_distance = max_distance
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        boundary_losses = []

        for c in range(target.shape[1]):
            gt = target[:, c:c + 1].float()
            prob = torch.sigmoid(pred[:, c:c + 1])

            kernel = torch.tensor(
                [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [1, -8, 1], [0, 1, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                dtype=torch.float32, device=pred.device
            ).unsqueeze(0).unsqueeze(0)

            boundary = torch.abs(F.conv3d(gt, kernel, padding=1))
            boundary = (boundary > 0.5).float()

            dist = torch.zeros_like(gt)
            for _ in range(int(self.max_distance)):
                dist = F.max_pool3d(dist + boundary, kernel_size=3, stride=1, padding=1)
                boundary = (dist > 0).float()

            dist = dist.clamp(0, self.max_distance)
            weight = torch.exp(-dist / self.sigma)
            loss_term = weight * (prob - gt).pow(2)
            boundary_losses.append(loss_term.mean())

        return torch.stack(boundary_losses).mean()


class CompoundSegLoss(nn.Module):
    def __init__(self, w_dice: float = 1.0, w_focal: float = 1.0, w_boundary: float = 0.5):
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
        self.boundary = DistanceBoundaryLoss(max_distance=5.0, sigma=2.0)
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_boundary = w_boundary

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_df = self.dice_focal(pred, target)
        l_bdry = self.boundary(pred, target)
        total = self.w_dice * l_df + self.w_boundary * l_bdry

        breakdown = {
            "dice_focal": l_df.item(),
            "boundary": l_bdry.item(),
            "total": total.item(),
        }
        return total, breakdown
