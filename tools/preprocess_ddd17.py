import sys
import os
# 把当前文件的上一级目录（也就是 MUSE-Net 根目录）加入到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from configs.config_ddd17 import config

OUTPUT_ROOT = Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification")

DATA_PATHS = {
    "train": {
        "images": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/aps_image"),
        "events": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/events"),
        "labels": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/labels/train"),
    },
    "test": {
        "images": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/rec1487417411_aps_images"),
        "events": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/rec1487417411_dvs_events_npy"),
        "labels": Path(r"/home/lizhuoxian/DataSets/dataset_DDD17-Events_our_codification/labels/test"),
    }
}

DDD17_H, DDD17_W = 260, 346
NUM_BINS = config.NUM_BINS


def generate_aet_representation(events_np, H, W):
    if events_np.shape[0] == 0:
        return torch.zeros((NUM_BINS * 2, H, W), dtype=torch.float32)

    t = events_np[:, 0].astype(int)
    x = events_np[:, 1].astype(int)
    y = events_np[:, 2].astype(float)
    p = events_np[:, 3].astype(int)

    valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t, p = x[valid_mask], y[valid_mask], t[valid_mask], p[valid_mask]

    if len(t) == 0:
        return torch.zeros((NUM_BINS * 2, H, W), dtype=torch.float32)

    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 0:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros_like(t)

    p = (p * 2 - 1) if p.min() >= 0 else p

    t_idx_float = t_norm * (NUM_BINS - 1)
    t_idx_floor = np.floor(t_idx_float).astype(int)
    t_idx_ceil = t_idx_floor + 1
    t_idx_ceil[t_idx_ceil >= NUM_BINS] = NUM_BINS - 1

    w_ceil = t_idx_float - t_idx_floor
    w_floor = 1.0 - w_ceil

    idx_floor = torch.tensor(t_idx_floor, dtype=torch.long)
    idx_ceil = torch.tensor(t_idx_ceil, dtype=torch.long)
    y_tens = torch.tensor(y, dtype=torch.long)
    x_tens = torch.tensor(x, dtype=torch.long)
    p_tens = torch.tensor(p, dtype=torch.float32)
    w_floor = torch.tensor(w_floor, dtype=torch.float32)
    w_ceil = torch.tensor(w_ceil, dtype=torch.float32)

    voxel_grid = torch.zeros((NUM_BINS, H, W), dtype=torch.float32)
    voxel_grid.index_put_((idx_floor, y_tens, x_tens), p_tens * w_floor, accumulate=True)
    voxel_grid.index_put_((idx_ceil, y_tens, x_tens), p_tens * w_ceil, accumulate=True)

    activity_map = torch.zeros((NUM_BINS, H, W), dtype=torch.float32)
    activity_map.index_put_((idx_floor, y_tens, x_tens), w_floor, accumulate=True)
    activity_map.index_put_((idx_ceil, y_tens, x_tens), w_ceil, accumulate=True)

    aet = torch.cat([voxel_grid, activity_map], dim=0)
    return aet


def process_dataset(split):
    paths = DATA_PATHS[split]
    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(list(paths["labels"].glob("*.png")))
    if not label_files:
        print(f"No labels found in {paths['labels']}")
        return

    print(f"Processing {split} set: {len(label_files)} samples -> AET (10 Channels)...")

    for label_path in tqdm(label_files):
        stem = label_path.stem

        img_candidates = list(paths["images"].glob(f"{stem}*"))
        if not img_candidates: continue
        img_path = img_candidates[0]

        evt_candidates = list(paths["events"].glob(f"{stem}*.npy"))
        if not evt_candidates: continue
        evt_path = evt_candidates[0]

        try:
            mask_np = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask_np is None: continue
            mask_t = torch.from_numpy(mask_np).long()

            rgb_np = cv2.imread(str(img_path))
            if rgb_np is None: continue
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)

            events_np = np.load(evt_path, allow_pickle=True)
            if events_np.dtype == np.object_:
                events_np = events_np.item()
            if len(events_np.shape) != 2 or events_np.shape[1] != 4:
                if events_np.shape[0] == 4: events_np = events_np.T

            aet_t = generate_aet_representation(events_np, DDD17_H, DDD17_W)
            aet_t = aet_t.half()

            data_dict = {
                "rgb": rgb_t,
                "event": aet_t,
                "mask": mask_t,
                "filename": stem
            }
            torch.save(data_dict, output_dir / f"{stem}.pt")

        except Exception as e:
            print(f"Error {stem}: {e}")


if __name__ == "__main__":
    process_dataset("train")
    process_dataset("test")