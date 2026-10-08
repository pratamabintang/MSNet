import torch
import numpy as np
import cv2
import os
from configs.config import config

# DSEC
COLOR_MAP_11 = np.array([
    [0, 0, 0],       # Background
    [70, 70, 70],    # Building
    [190, 153, 153], # Fence
    [220, 20, 60],   # Person
    [153, 153, 153], # Pole
    [128, 64, 128],  # Road
    [244, 35, 232],  # Sidewalk
    [107, 142, 35],  # Vegetation
    [0, 0, 142],     # Car
    [102, 102, 156], # Wall
    [220, 220, 0]    # TrafficSign
], dtype=np.uint8)

COLOR_MAP_11 = COLOR_MAP_11[:, ::-1]

COLOR_MAP_DDD17 = np.array([
    [128, 64, 128], [70, 70, 70], [153, 153, 153],
    [107, 142, 35], [220, 20, 60], [0, 0, 142]
], dtype=np.uint8)

COLOR_MAP_DDD17 = COLOR_MAP_DDD17[:, ::-1]

# Cityscapes
COLOR_MAP_19 = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)[:, ::-1]


def get_color_map(num_classes=None):
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if num_classes == 11:
        return COLOR_MAP_11
    elif num_classes == 6:
        return COLOR_MAP_DDD17
    else:
        return COLOR_MAP_19


def colorize_mask(mask_np, num_classes=None):
    # Infer the class count from the label map when not given, so DDD17
    # (6 classes) and DSEC (11 classes) are colored with their own palette.
    if num_classes is None:
        num_classes = int(mask_np.max()) + 1
    cmap = get_color_map(num_classes)
    mask_np = np.clip(mask_np, 0, len(cmap) - 1)
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for cid in range(len(cmap)):
        color_mask[mask_np == cid] = cmap[cid]
    return color_mask

class RunningConfusionMatrix:

    def __init__(self, num_classes, ignore_label=255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        # pred: (B, H, W), target: (B, H, W)
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        mask = (target != self.ignore_label) & (target < self.num_classes)

        pred = pred[mask]
        target = target[mask]

        if len(pred) == 0:
            return

        indices = target * self.num_classes + pred
        m = np.bincount(indices, minlength=self.num_classes ** 2)

        if len(m) > self.num_classes ** 2:
            m = m[:self.num_classes ** 2]

        m = m.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m

    def compute(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        intersection = tp
        union = tp + fp + fn

        with np.errstate(divide='ignore', invalid='ignore'):
            ious = intersection / union
            ious[union == 0] = np.nan

        miou = np.nanmean(ious)
        return miou, ious

    def reset(self):
        self.confusion_matrix.fill(0)


def compute_miou(logits, targets):
    preds = torch.argmax(logits, dim=1)
    if targets.device != preds.device:
        targets = targets.to(preds.device)
    num_classes = config.NUM_CLASSES
    class_ious = []
    for c in range(num_classes):
        pred_mask = (preds == c)
        target_mask = (targets == c)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            class_ious.append(float('nan'))
        else:
            class_ious.append((intersection / union).item())
    valid_ious = [x for x in class_ious if not np.isnan(x)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    print_ious = [0.0 if np.isnan(x) else x for x in class_ious]
    return miou, print_ious

def save_debug_images(rgb, event, mask, logits, step, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    # RGB
    rgb_img = (rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Pred
    if isinstance(logits, tuple): logits = logits[0]
    pred = torch.argmax(logits[0], dim=0).detach().cpu().numpy()
    pred_color = colorize_mask(pred)

    # GT
    gt = mask[0].detach().cpu().numpy()
    gt_color = colorize_mask(gt)

    # Resize GT/Pred to Match RGB
    h, w = rgb_img.shape[:2]
    if gt_color.shape[:2] != (h, w):
        gt_color = cv2.resize(gt_color, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_color.shape[:2] != (h, w):
        pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)

    vis = np.hstack([rgb_img, gt_color, pred_color])
    cv2.imwrite(os.path.join(save_dir, f"step_{step}.jpg"), vis)

def generate_aet_representation(events, shape, nr_temporal_bins=5):

    height, width = shape

    if len(events) == 0:
        return np.zeros((2 * nr_temporal_bins, height, width), dtype=np.float32)

    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp
    if deltaT == 0: deltaT = 1.0

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(np.int32)
    dts = ts - tis

    vals_left = np.abs(pols) * (1.0 - dts) * np.sign(pols)
    vals_right = np.abs(pols) * dts * np.sign(pols)

    voxel_grid = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    valid_mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height) & (tis >= 0) & (tis < nr_temporal_bins)
    flat_indices = xs[valid_mask] + ys[valid_mask] * width + tis[valid_mask] * width * height
    np.add.at(voxel_grid, flat_indices, vals_left[valid_mask])

    valid_mask_next = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height) & ((tis + 1) < nr_temporal_bins)
    flat_indices_next = xs[valid_mask_next] + ys[valid_mask_next] * width + (tis[valid_mask_next] + 1) * width * height
    np.add.at(voxel_grid, flat_indices_next, vals_right[valid_mask_next])

    voxel_grid = voxel_grid.reshape((nr_temporal_bins, height, width))

    map_grid = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()

    count_left = (1.0 - dts)
    count_right = dts

    np.add.at(map_grid, flat_indices, count_left[valid_mask])
    np.add.at(map_grid, flat_indices_next, count_right[valid_mask_next])

    map_grid = map_grid.reshape((nr_temporal_bins, height, width))

    return np.concatenate((voxel_grid, map_grid), axis=0)