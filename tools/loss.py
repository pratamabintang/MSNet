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


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Structure loss: Weighted Binary Cross-Entropy + Weighted IoU.

    Applies boundary-focused weighting using average pooling deviation.
    """
    pred = pred.float()
    mask = mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def compute_iou(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes binary IoU score between prediction and ground truth."""
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter = (pred_bin * mask).sum().item()
    union = (pred_bin + mask).clamp(0, 1).sum().item()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + 1e-7)


def compute_metrics(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> dict:
    """Computes comprehensive segmentation metrics (IoU, F1/Dice, Precision, Recall)."""
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter = (pred_bin * mask).sum().item()
    union = (pred_bin + mask).clamp(0, 1).sum().item()
    iou = inter / (union + 1e-7) if union > 0 else (1.0 if inter == 0 else 0.0)

    tp = inter
    fp = (pred_bin * (1.0 - mask)).sum().item()
    fn = ((1.0 - pred_bin) * mask).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }