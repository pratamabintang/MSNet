"""Reusable building blocks for MUSE-Net.

The modules in this file are intentionally small and composable.  They are
shared by the DSEC model and the DDD17 model.  Public class names are kept
stable so existing checkpoints and training scripts can continue to import
the same symbols.

Mapping to the paper's module names:
    DSEM                      -> Density-guided Spatiotemporal Event Modulation
    LocalTopoGraphConv        -> SDGSR (Sparse-Dense Graph Synergistic Representation)
    DynamicRouting            -> UADR (Uncertainty-Aware Dynamic Routing)
    DeformableCrossAttention  -> Deformable Cross-Attention (MSFE spatial branch)
    CrossModalSSMFusion       -> CM-SE (Cross-Modal State Evolution, MSFE temporal branch)
    CrossModalSpectralFusion  -> Cross-Modal Spectral Fusion (MSFE frequency branch)
    EventBoundaryRefinementModule -> EBRM (Event Boundary Refinement Module)
    network.muse_net.FusionBlock -> MSFE (Multi-dimensional Synergistic Fusion Engine)
    tools.loss.CrossModalContrastiveLoss -> CM-SECA (Cross-Modal Semantic Contrastive Alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSEM(nn.Module):
    """Density-guided Spatiotemporal Event Modulation.

    Expected inputs:
        ev:       event voxel tensor with shape [B, T, H, W]
        map_data: activity/density tensor with shape [B, T, H, W]

    DSEM learns two residual corrections:
        1. Spatial affine modulation from the event activity map.
        2. Temporal-bin mixing implemented as ordinary 2D convolutions over
           the bin dimension, which keeps the module lightweight.

    The learnable residual branches are zero-initialized.  At initialization
    the module is therefore close to an identity mapping, which makes it safer
    to insert into a pretrained pipeline.
    """

    def __init__(self, in_dim=1, out_dim=64, num_bins=5):
        super().__init__()
        self.in_dim = in_dim
        self.num_bins = num_bins

        # Extract spatial density cues from each time bin independently.
        self.density_extractor = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

        # Gamma and beta form an affine correction for the event voxel grid.
        self.gamma_proj = nn.Conv2d(out_dim, 1, kernel_size=3, padding=1)
        self.beta_proj = nn.Conv2d(out_dim, 1, kernel_size=3, padding=1)

        # Keep the initial behavior close to ev -> ev.
        nn.init.constant_(self.gamma_proj.weight, 0)
        nn.init.constant_(self.gamma_proj.bias, 0)
        nn.init.constant_(self.beta_proj.weight, 0)
        nn.init.constant_(self.beta_proj.bias, 0)

        # A depthwise spatial filter followed by pointwise bin mixing.
        self.temporal_mixing = nn.Sequential(
            nn.Conv2d(num_bins, num_bins, kernel_size=3, padding=1, groups=num_bins, bias=False),
            nn.BatchNorm2d(num_bins),
            nn.GELU(),
            nn.Conv2d(num_bins, num_bins, kernel_size=1),
        )

        # The residual temporal branch also starts from zero.
        nn.init.constant_(self.temporal_mixing[-1].weight, 0)
        nn.init.constant_(self.temporal_mixing[-1].bias, 0)

    def forward(self, ev, map_data):
        batch_size, num_bins, height, width = ev.shape

        # Run the density extractor on each temporal bin as an independent
        # image.  With in_dim=1, [B, T, H, W] becomes [B*T, 1, H, W].
        map_reshaped = map_data.reshape(batch_size * num_bins, self.in_dim, height, width)
        density_feat = self.density_extractor(map_reshaped)

        gamma = self.gamma_proj(density_feat).reshape(batch_size, num_bins, height, width)
        beta = self.beta_proj(density_feat).reshape(batch_size, num_bins, height, width)

        # Bound gamma to [-1, 1].  The multiplicative factor (1 + gamma) is
        # therefore in [0, 2], which is friendly to AMP/FP16 training.
        gamma = torch.tanh(gamma)
        ev_spatial = ev * (1.0 + gamma) + beta

        # Mix information across nearby pixels and across temporal bins.
        ev_temporal = self.temporal_mixing(ev_spatial)
        return ev_spatial + ev_temporal


class DeformableCrossAttention(nn.Module):
    """Lightweight deformable cross-attention from event queries to RGB keys.

    Event features produce the query and RGB features produce keys/values.  The
    RGB sampling locations are predicted from the concatenated RGB-event pair.
    The offset predictor is zero-initialized, so the first forward pass samples
    from regular grid locations.
    """

    def __init__(self, channels, num_heads=8, num_points=9):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.num_points = num_points
        self.channels = channels
        self.head_dim = channels // num_heads

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_heads * num_points * 2, 3, padding=1),
        )
        nn.init.constant_(self.offset_conv[2].weight, 0)
        nn.init.constant_(self.offset_conv[2].bias, 0)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _get_reference_points(height, width, batch_size, device):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        ref_y = y.float() / (height - 1)
        ref_x = x.float() / (width - 1)
        return torch.stack((ref_x, ref_y), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward(self, rgb, evt):
        batch_size, channels, height, width = evt.shape

        q = self.q_proj(evt).view(batch_size, self.num_heads, self.head_dim, height, width)

        # Offsets are bounded to a quarter of the normalized image range.  This
        # keeps early training stable while still allowing local deformation.
        combined = torch.cat([evt, rgb], dim=1)
        offsets = self.offset_conv(combined)
        offsets = offsets.view(batch_size, self.num_heads, self.num_points, 2, height, width)
        offsets = torch.tanh(offsets) * 0.25

        ref_points = self._get_reference_points(height, width, batch_size, evt.device).unsqueeze(1).unsqueeze(1)
        sampling_locations = ref_points + offsets
        sampling_grid = sampling_locations * 2.0 - 1.0
        sampling_grid = sampling_grid.permute(0, 1, 4, 5, 2, 3).flatten(0, 1).flatten(2, 3)

        k_in = self.k_proj(rgb).view(batch_size * self.num_heads, self.head_dim, height, width)
        v_in = self.v_proj(rgb).view(batch_size * self.num_heads, self.head_dim, height, width)

        k_sampled = F.grid_sample(k_in, sampling_grid, align_corners=False)
        v_sampled = F.grid_sample(v_in, sampling_grid, align_corners=False)

        k_sampled = k_sampled.view(batch_size, self.num_heads, self.head_dim, height, width, self.num_points)
        v_sampled = v_sampled.view(batch_size, self.num_heads, self.head_dim, height, width, self.num_points)

        attn_logits = (q.unsqueeze(-1) * k_sampled).sum(dim=2)
        attn_weights = F.softmax(attn_logits * self.scale, dim=-1).unsqueeze(2)
        attended = (attn_weights * v_sampled).sum(dim=-1).reshape(batch_size, channels, height, width)

        return evt + self.out_proj(attended)


class EventBoundaryRefinementModule(nn.Module):
    """Refine semantic logits with a high-resolution event boundary gate."""

    def __init__(self, event_channels, num_classes):
        super().__init__()
        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(event_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        # Start as a neutral residual gate.
        nn.init.constant_(self.boundary_extractor[-1].weight, 0)
        if self.boundary_extractor[-1].bias is not None:
            nn.init.constant_(self.boundary_extractor[-1].bias, 0)

    def forward(self, semantic_logits, event_low_level_feat):
        event_low_level_feat = F.interpolate(
            event_low_level_feat,
            size=semantic_logits.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_gate = torch.sigmoid(self.boundary_extractor(event_low_level_feat))
        return semantic_logits + semantic_logits * boundary_gate


class CrossModalSpectralFusion(nn.Module):
    """Frequency-amplitude guided residual fusion.

    FFT operations are evaluated in float32 even during mixed precision
    training because complex FFT kernels are numerically sensitive in FP16.
    The learned gate is cast back to the original feature dtype afterward.
    """

    def __init__(self, dim):
        super().__init__()
        self.spectral_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

        nn.init.constant_(self.spectral_gate[-1].weight, 0)
        nn.init.constant_(self.spectral_gate[-1].bias, 0)

    def forward(self, rgb, evt):
        height, width = rgb.shape[2], rgb.shape[3]
        orig_dtype = rgb.dtype

        rgb_fft = torch.fft.rfft2(rgb.to(torch.float32), norm="ortho")
        evt_fft = torch.fft.rfft2(evt.to(torch.float32), norm="ortho")

        rgb_struct = torch.fft.irfft2(torch.abs(rgb_fft), s=(height, width), norm="ortho")
        evt_struct = torch.fft.irfft2(torch.abs(evt_fft), s=(height, width), norm="ortho")

        gate = torch.sigmoid(self.spectral_gate(torch.cat([rgb_struct, evt_struct], dim=1)))
        gate = gate.to(orig_dtype)
        return rgb + rgb * gate


class LocalTopoGraphConv(nn.Module):
    """Local topology aggregation over a 3x3 event neighborhood.

    The module builds edge features from each center pixel and its 3x3
    neighbors, aggregates them with a shared 1x1 convolution, and returns a
    zero-initialized residual update.
    """

    def __init__(self, in_channels, k_neighbors=9):
        super().__init__()
        if k_neighbors != 9:
            raise ValueError("LocalTopoGraphConv currently expects k_neighbors=9 for a 3x3 neighborhood.")

        self.k = k_neighbors
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.constant_(self.proj.weight, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # F.unfold extracts the 3x3 neighborhood around every spatial position.
        unfolded = F.unfold(x, kernel_size=3, padding=1)
        unfolded = unfolded.view(batch_size, channels, self.k, height, width)

        center = x.unsqueeze(2).expand(-1, -1, self.k, -1, -1)
        edge_feat = torch.cat([unfolded - center, center], dim=1)
        edge_feat = edge_feat.view(batch_size, channels * 2 * self.k, height, width)

        # Process each neighbor slot with shared weights, then average.
        agg_feat = self.edge_conv(edge_feat.view(batch_size * self.k, channels * 2, height, width))
        agg_feat = agg_feat.view(batch_size, self.k, channels, height, width).mean(dim=1)
        return x + self.proj(agg_feat)


class UncertaintyEstimator(nn.Module):
    """Predict a per-pixel uncertainty map for one modality."""

    def __init__(self, dim):
        super().__init__()
        hidden_dim = max(dim // 4, 1)
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.uncertainty_net(x))


class DynamicRouting(nn.Module):
    """Uncertainty-aware dynamic routing for RGB-event feature pairs."""

    def __init__(self, dim):
        super().__init__()
        self.u_rgb = UncertaintyEstimator(dim)
        self.u_evt = UncertaintyEstimator(dim)

    def forward(self, rgb, evt):
        uncert_rgb = self.u_rgb(rgb)
        uncert_evt = self.u_evt(evt)

        # Lower uncertainty means higher reliability.  Normalize the two
        # reliability maps so the routed features stay on a comparable scale.
        weight_rgb = torch.exp(-uncert_rgb)
        weight_evt = torch.exp(-uncert_evt)
        weight_sum = weight_rgb + weight_evt + 1e-6

        return rgb * (weight_rgb / weight_sum), evt * (weight_evt / weight_sum)


class CrossModalSSMFusion(nn.Module):
    """Event-conditioned state-space style fusion.

    This module flattens each feature map into a sequence and uses event
    features to control how RGB state is updated.  It is intentionally compact:
    the operation is an efficient selective-scan approximation rather than a
    full recurrent SSM implementation.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.state_proj_rgb = nn.Conv1d(dim, dim, 1)
        self.state_proj_evt = nn.Conv1d(dim, dim, 1)

        self.dt_proj = nn.Sequential(
            nn.Conv1d(dim, dim // 4, 1),
            nn.SiLU(),
            nn.Conv1d(dim // 4, dim, 1),
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, rgb, evt):
        batch_size, channels, height, width = rgb.shape

        seq_rgb = rgb.view(batch_size, channels, -1)
        seq_evt = evt.view(batch_size, channels, -1)

        rgb_state = self.state_proj_rgb(seq_rgb)
        evt_state = self.state_proj_evt(seq_evt)

        # Positive event-conditioned control signal.
        dt = F.softplus(self.dt_proj(evt_state))
        state = rgb_state * torch.exp(-dt) + evt_state * dt

        state = state.view(batch_size, channels, height, width)
        return rgb + self.out_proj(state)


__all__ = [
    "CrossModalSSMFusion",
    "CrossModalSpectralFusion",
    "DSEM",
    "DeformableCrossAttention",
    "DynamicRouting",
    "EventBoundaryRefinementModule",
    "LocalTopoGraphConv",
    "UncertaintyEstimator",
]
