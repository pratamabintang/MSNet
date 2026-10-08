"""Landslide model entry point.

Wraps LandslideMUSENet for RGB + LiDAR Topography semantic segmentation.
"""

from configs.config_landslide import config
from network.muse_net import (
    AdvancedMambaRoutingFusion,
    LandslideMUSENet,
    MLP,
    SegFormerHead,
)


class EGHFNet(LandslideMUSENet):
    """MUSE-Net configured for Landslide Segmentation (RGB + LiDAR Topography)."""

    def __init__(self, topo_in_channels=None, num_classes=None, use_graph=None):
        in_chans = config.TOPO_INPUT_CHANS if topo_in_channels is None else topo_in_channels
        classes = config.NUM_CLASSES if num_classes is None else num_classes
        graph = config.USE_GRAPH if use_graph is None else use_graph

        super().__init__(
            topo_in_channels=in_chans,
            num_classes=classes,
            use_graph=graph,
            rgb_backbone=getattr(config, "RGB_BACKBONE", "b2"),
            topo_backbone=getattr(config, "TOPO_BACKBONE", "b0"),
            pretrained_rgb_path=getattr(config, "PRETRAINED_RGB_PATH", None),
            pretrained_topo_path=getattr(config, "PRETRAINED_TOPO_PATH", None),
        )


__all__ = [
    "AdvancedMambaRoutingFusion",
    "EGHFNet",
    "LandslideMUSENet",
    "MLP",
    "SegFormerHead",
]
