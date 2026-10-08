"""DSEC model entry point.

Training and testing scripts import ``EGHFNet`` from this file.  The actual
architecture lives in ``network.muse_net`` so the DSEC and DDD17 variants share
one clean implementation.
"""

from configs.config import config
from network.muse_net import (
    AdvancedMambaRoutingFusion,
    DSECMUSENet,
    MLP,
    SegFormerHead,
)


class EGHFNet(DSECMUSENet):
    """MUSE-Net configured for the DSEC setup in ``configs/config.py``."""

    def __init__(self):
        super().__init__(
            num_classes=config.NUM_CLASSES,
            event_input_channels=config.EVENT_INPUT_CHANS,
        )


__all__ = [
    "AdvancedMambaRoutingFusion",
    "EGHFNet",
    "MLP",
    "SegFormerHead",
]
