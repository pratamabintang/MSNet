"""DDD17 model entry point.

This wrapper keeps the original public API while delegating the shared network
logic to ``network.muse_net``.
"""

from configs.config_ddd17 import config
from network.muse_net import (
    AdvancedMambaRoutingFusion,
    DDD17MUSENet,
    MLP,
    SegFormerHead,
)


class EGHFNet(DDD17MUSENet):
    """MUSE-Net configured for DDD17 grayscale RGB-event segmentation."""

    def __init__(self, num_classes=6, num_bins=None):
        super().__init__(
            num_classes=num_classes,
            num_bins=config.NUM_BINS if num_bins is None else num_bins,
        )


__all__ = [
    "AdvancedMambaRoutingFusion",
    "EGHFNet",
    "MLP",
    "SegFormerHead",
]
