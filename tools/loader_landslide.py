import torch
from torch.utils.data import DataLoader
from configs.config_landslide import config
from dataset import LandslideDataset
from typing import Optional, List


class LandslideCollate:
    """Top-level collate callable compatible with Windows multiprocessing."""

    def __init__(self, mode_flag: str = "real"):
        self.mode_flag = str(mode_flag).lower()

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch], dim=0)
        labels = torch.stack([item["label"] for item in batch], dim=0)
        names = [item["name"] for item in batch]

        # Extract RGB (first 3 channels)
        rgb = images[:, :3, :, :]

        # Extract Topography (remaining channels)
        if images.shape[1] > 3:
            topo = images[:, 3:, :, :]
        else:
            # Fallback if only IMAGE is loaded
            topo = torch.zeros((images.shape[0], 1, images.shape[2], images.shape[3]), dtype=torch.float32)

        # Handle 'zero' mode (DTM 0% ablation control)
        # Adding 1e-6 epsilon prevents degenerate zero-variance in SegFormer LayerNorm backward
        if self.mode_flag == "zero":
            topo = torch.zeros_like(topo) + 1e-6

        return {
            "rgb": rgb,
            "topo": topo,
            "mask": labels,
            "name": names,
        }


def get_landslide_dataloaders(
    split: str = "train",
    batch_size: Optional[int] = None,
    modalities: Optional[List[str]] = None,
    run_mode: Optional[str] = None,
    num_workers: int = 0,
    size: Optional[int] = None,
    skip_shape_check: bool = True,
    data_dir: Optional[str] = None,
    blacklist_path: Optional[str] = None,
    split_paths: Optional[dict] = None,
):
    """Creates a PyTorch DataLoader for the multi-modal Landslide dataset.

    Separates the composite raster tensor from `LandslideDataset` into:
        - `rgb`: Optical RGB channels [B, 3, H, W]
        - `topo`: LiDAR topography channels [B, C_topo, H, W]
        - `mask`: Ground truth binary label mask [B, 1, H, W]
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    mods = modalities if modalities is not None else config.MODALITIES
    target_size = size if size is not None else config.IMG_SIZE
    mode_flag = run_mode if run_mode is not None else getattr(config, "RUN_MODE", "real")
    d_dir = data_dir if data_dir is not None else str(config.DATA_DIR)
    b_path = blacklist_path if blacklist_path is not None else str(config.BLACKLIST_PATH)

    dataset = LandslideDataset(
        data_dir=d_dir,
        split=split,
        size=target_size,
        modalities=mods,
        blacklist_path=b_path,
        skip_shape_check=skip_shape_check,
        split_paths=split_paths,
    )

    is_train = split == "train"
    collate = LandslideCollate(mode_flag=mode_flag)

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
        collate_fn=collate,
    )

    return loader
