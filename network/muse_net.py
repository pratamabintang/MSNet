"""Clean MUSE-Net model definitions.

This file contains the shared architecture used by the dataset-specific entry
points in ``network/model.py`` and ``network/model_DDD17.py``.  The wrappers
keep the original public class name ``EGHFNet`` so existing training scripts
and checkpoints remain compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.backbone import DualSegFormerBackbone
from network.modules import (
    CrossModalSpectralFusion,
    CrossModalSSMFusion,
    DSEM,
    DeformableCrossAttention,
    DynamicRouting,
    EventBoundaryRefinementModule,
    LocalTopoGraphConv,
)


class FusionBlock(nn.Module):
    """Fuse one RGB feature level with the matching event feature level.

    Each block follows the same three-step pattern:
        1. Project event channels to the RGB feature dimension.
        2. Route RGB/event features with uncertainty-aware weights.
        3. Fuse the routed pair through spatial, sequential, and spectral
           branches, then compress the concatenated result.

    The returned routed features are used only during training for the
    cross-modal contrastive loss.
    """

    def __init__(self, dim_rgb, dim_evt, num_heads):
        super().__init__()
        self.dim_rgb = dim_rgb

        if dim_evt != dim_rgb:
            self.proj_evt = nn.Sequential(
                nn.Conv2d(dim_evt, dim_rgb, 1, bias=False),
                nn.BatchNorm2d(dim_rgb),
            )
        else:
            self.proj_evt = nn.Identity()

        self.router = DynamicRouting(dim_rgb)
        self.deform_attn = DeformableCrossAttention(dim_rgb, num_heads=num_heads)
        self.ssm_fusion = CrossModalSSMFusion(dim_rgb)
        self.spectral_fusion = CrossModalSpectralFusion(dim_rgb)

        self.proj = nn.Sequential(
            nn.Conv2d(dim_rgb * 3, dim_rgb, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_rgb),
            nn.ReLU(True),
        )

    def forward(self, rgb, evt):
        evt = self.proj_evt(evt)
        rgb_routed, evt_routed = self.router(rgb, evt)

        feat_spatial = self.deform_attn(rgb_routed, evt_routed)
        feat_sequence = self.ssm_fusion(rgb_routed, evt_routed)
        feat_spectral = self.spectral_fusion(rgb_routed, evt_routed)

        fused = torch.cat([feat_spatial, feat_sequence, feat_spectral], dim=1)
        return self.proj(fused), rgb_routed, evt_routed


class FeatureProjectionMLP(nn.Module):
    """SegFormer-style per-pixel linear projection.

    Input shape:  [B, C, H, W]
    Output shape: [B, H*W, embed_dim]
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class SegFormerHead(nn.Module):
    """Lightweight decoder head used after multi-scale fusion."""

    def __init__(self, in_channels_list, embedding_dim=256, num_classes=11):
        super().__init__()
        c1, c2, c3, c4 = in_channels_list

        self.linear_c4 = FeatureProjectionMLP(input_dim=c4, embed_dim=embedding_dim)
        self.linear_c3 = FeatureProjectionMLP(input_dim=c3, embed_dim=embedding_dim)
        self.linear_c2 = FeatureProjectionMLP(input_dim=c2, embed_dim=embedding_dim)
        self.linear_c1 = FeatureProjectionMLP(input_dim=c1, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def _project_and_resize(self, feat, projection, target_size):
        batch_size, _, height, width = feat.shape
        feat = projection(feat).permute(0, 2, 1).reshape(batch_size, -1, height, width)
        if feat.shape[2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        return feat

    def forward(self, x1, x2, x3, x4):
        target_size = x1.shape[2:]
        c4 = self._project_and_resize(x4, self.linear_c4, target_size)
        c3 = self._project_and_resize(x3, self.linear_c3, target_size)
        c2 = self._project_and_resize(x2, self.linear_c2, target_size)
        c1 = self._project_and_resize(x1, self.linear_c1, target_size)

        fused = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        return self.classifier(self.dropout(fused))


class MUSEBackboneDecoder(nn.Module):
    """Shared RGB-event backbone, fusion stages, decoder, and boundary refine."""

    rgb_chans = [64, 128, 320, 512]
    evt_chans = [32, 64, 160, 256]

    def __init__(self, event_backbone_channels, num_classes):
        super().__init__()
        self.backbone = DualSegFormerBackbone(event_in_channels=event_backbone_channels)

        self.fusion1 = FusionBlock(self.rgb_chans[0], self.evt_chans[0], num_heads=4)
        self.fusion2 = FusionBlock(self.rgb_chans[1], self.evt_chans[1], num_heads=4)
        self.fusion3 = FusionBlock(self.rgb_chans[2], self.evt_chans[2], num_heads=8)
        self.fusion4 = FusionBlock(self.rgb_chans[3], self.evt_chans[3], num_heads=8)

        self.decoder = SegFormerHead(
            in_channels_list=self.rgb_chans,
            embedding_dim=256,
            num_classes=num_classes,
        )
        self.ebrm = EventBoundaryRefinementModule(
            event_channels=self.evt_chans[0],
            num_classes=num_classes,
        )

    def _forward_backbone_decoder(self, rgb, evt, output_size):
        """Run the common feature extraction, fusion, decoding, and refinement."""

        rgb_feats, evt_feats = self.backbone(rgb, evt)

        fused1, _, _ = self.fusion1(rgb_feats[0], evt_feats[0])
        fused2, _, _ = self.fusion2(rgb_feats[1], evt_feats[1])
        fused3, _, _ = self.fusion3(rgb_feats[2], evt_feats[2])
        fused4, rgb_stage4, evt_stage4 = self.fusion4(rgb_feats[3], evt_feats[3])

        logits = self.decoder(fused1, fused2, fused3, fused4)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        logits = self.ebrm(logits, evt_feats[0])

        if self.training:
            return logits, [(rgb_stage4, evt_stage4)]
        return logits


class DSECMUSENet(MUSEBackboneDecoder):
    """MUSE-Net entry point for DSEC-style RGB-event segmentation.

    DSEC preprocessing in this project usually provides ``EVENT_INPUT_CHANS``
    channels directly to the event backbone.  If a paired voxel/activity tensor
    with twice that channel count is provided, DSEM and SDGSR are applied first.
    """

    def __init__(self, num_classes, event_input_channels):
        self.evt_in_chans = event_input_channels
        super().__init__(event_backbone_channels=event_input_channels, num_classes=num_classes)

        self.aeim = DSEM(in_dim=1, out_dim=64, num_bins=self.evt_in_chans)
        self.sdgsr = LocalTopoGraphConv(in_channels=self.evt_in_chans, k_neighbors=9)

    def forward(self, rgb, evt):
        if evt.shape[1] == 2 * self.evt_in_chans:
            event_voxels = evt[:, :self.evt_in_chans, :, :]
            activity_maps = evt[:, self.evt_in_chans:, :, :]
            evt = self.aeim(event_voxels, activity_maps)
            evt = self.sdgsr(evt)

        return self._forward_backbone_decoder(rgb, evt, output_size=rgb.shape[2:])


class DDD17MUSENet(MUSEBackboneDecoder):
    """MUSE-Net entry point for DDD17 grayscale RGB-event segmentation."""

    def __init__(self, num_classes, num_bins):
        self.num_bins = num_bins
        super().__init__(event_backbone_channels=3, num_classes=num_classes)

        self.aeim = DSEM(in_dim=1, out_dim=64, num_bins=self.num_bins)
        self.sdgsr = LocalTopoGraphConv(in_channels=self.num_bins, k_neighbors=9)

    @staticmethod
    def _select_three_event_channels(evt):
        """Map variable event-bin counts to the 3 channels expected by MiT-B0."""

        channels = evt.shape[1]
        if channels == 1:
            return evt.repeat(1, 3, 1, 1)
        if channels == 5:
            return evt[:, [0, 2, 4], :, :]
        if channels == 7:
            return evt[:, [0, 3, 6], :, :]
        if channels > 3:
            return evt[:, :3, :, :]
        return evt

    def forward(self, rgb, evt):
        rgb = rgb.repeat(1, 3, 1, 1) if rgb.shape[1] == 1 else rgb

        if evt.shape[1] == 2 * self.num_bins:
            event_voxels = evt[:, :self.num_bins, :, :]
            activity_maps = evt[:, self.num_bins:, :, :]
            evt = self.aeim(event_voxels, activity_maps)
            evt = self.sdgsr(evt)
        else:
            evt = self.sdgsr(evt[:, :self.num_bins, :, :])

        evt = self._select_three_event_channels(evt)
        return self._forward_backbone_decoder(rgb, evt, output_size=rgb.shape[2:])


# Backward-compatible aliases used by older scripts.
AdvancedMambaRoutingFusion = FusionBlock
MLP = FeatureProjectionMLP


__all__ = [
    "AdvancedMambaRoutingFusion",
    "DDD17MUSENet",
    "DSECMUSENet",
    "FeatureProjectionMLP",
    "FusionBlock",
    "MLP",
    "MUSEBackboneDecoder",
    "SegFormerHead",
]
