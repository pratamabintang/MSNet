import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import config
from tools.lovasz_losses import LovaszSoftmax

class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, min_kept=20000, ignore_index=255, weights=None, keep_ratio=0.2):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(config.DEVICE)
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.keep_ratio = keep_ratio
        self.update_weights(weights)

    def update_weights(self, weights):
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            weights = weights.float().to(config.DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index, reduction='none')

    def forward(self, logits, target):
        pixel_losses = self.ce(logits, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index
        valid_losses = pixel_losses[mask]

        if len(valid_losses) == 0:
            return torch.tensor(0.0).to(logits.device)
        num_valid = valid_losses.numel()
        keep_num = max(int(num_valid * self.keep_ratio), self.min_kept)
        keep_num = min(keep_num, num_valid)
        topk_loss, _ = valid_losses.topk(keep_num)
        return topk_loss.mean()

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, rgb_feat, evt_feat):
        B, C, H, W = rgb_feat.shape
        rgb_flat = F.normalize(rgb_feat.view(B, C, -1), dim=1)
        evt_flat = F.normalize(evt_feat.view(B, C, -1), dim=1)

        sim_matrix = torch.bmm(rgb_flat.transpose(1, 2), evt_flat) / self.temp
        labels = torch.arange(H * W, device=rgb_feat.device).unsqueeze(0).expand(B, -1)

        loss_r2e = F.cross_entropy(sim_matrix.view(-1, H * W), labels.flatten())
        loss_e2r = F.cross_entropy(sim_matrix.transpose(1, 2).reshape(-1, H * W), labels.flatten())
        return (loss_r2e + loss_e2r) * 0.5