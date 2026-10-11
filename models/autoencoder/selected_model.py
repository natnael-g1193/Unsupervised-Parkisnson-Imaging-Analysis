import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def _compute_ladder_shapes(initial_shape, target_shape, stages=4):
    """Plan intermediate spatial sizes so the ladder lands exactly on `target_shape`."""
    current = tuple(initial_shape)
    target = tuple(target_shape)
    shapes = []

    for stage in range(stages):
        remaining = stages - stage - 1
        next_dims = []
        for dim_current, dim_target in zip(current, target):
            if dim_current == dim_target or remaining < 0:
                next_dims.append(dim_target)
                continue

            if remaining == 0:
                next_dims.append(dim_target)
                continue

            min_dim = math.ceil(dim_target / (2 ** remaining))
            proposed = dim_current * 2
            next_dims.append(min(dim_target, max(proposed, min_dim)))

        current = tuple(next_dims)
        shapes.append(current)

    if shapes:
        shapes[-1] = target
    return shapes


class BaseConvBlock(nn.Module):
    """3D convolution followed by batch normalization and inplace ReLU for compact feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))  # inplace ReLU for memory efficiency
        ]))

    def forward(self, x):
        return self.block(x)


class BaseChannelReductionBlock(nn.Module):
    """Channel compressor that applies a 1×1×1 convolution before normalization and activation."""
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()
        self.out_channels = in_channels // reduction_factor
        self.block = nn.Sequential(OrderedDict([
            ('conv1x1', nn.Conv3d(in_channels, self.out_channels, kernel_size=1)),
            ('bn', nn.BatchNorm3d(self.out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    
    def forward(self, x):
        return self.block(x)


class BaseEncoder(nn.Module):
    """Four-stage 3D encoder that downsamples (1, 64, 128, 128) volumes to a latent tensor of shape (128, 4, 8, 8)."""
    def __init__(self, initial_filters=4):
        super().__init__()
        
        # Stage 1: spatial stride 2, expand to 32 channels, then compress to 8
        # Input tensor: (1, 64, 128, 128)
        self.down1 = nn.Sequential(
            BaseConvBlock(1, initial_filters, stride=2),      # -> (4, 32, 64, 64)
            BaseConvBlock(initial_filters, 32),               # -> (32, 32, 64, 64)
            BaseChannelReductionBlock(32)                     # -> (8, 32, 64, 64)
        )
        
        # Stage 2: repeat downsampling while keeping 8-channel output
        # Input tensor: (8, 32, 64, 64)
        self.down2 = nn.Sequential(
            BaseConvBlock(8, 32, stride=2),                   # -> (32, 16, 32, 32)
            BaseConvBlock(32, 32),                            # -> (32, 16, 32, 32)
            BaseChannelReductionBlock(32)                     # -> (8, 16, 32, 32)
        )
        
        # Stage 3: widen to 64 features before compressing to 16 channels
        # Input tensor: (8, 16, 32, 32)
        self.down3 = nn.Sequential(
            BaseConvBlock(8, 64, stride=2),                   # -> (64, 8, 16, 16)
            BaseConvBlock(64, 64),                            # -> (64, 8, 16, 16)
            BaseChannelReductionBlock(64)                     # -> (16, 8, 16, 16)
        )
        
        # Stage 4: reach latent representation without channel reduction
        # Input tensor: (16, 8, 16, 16)
        self.down4 = nn.Sequential(
            BaseConvBlock(16, 128, stride=2),                 # -> (128, 4, 8, 8)
            BaseConvBlock(128, 128),                          # -> (128, 4, 8, 8)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4


class BottleneckEncoder(nn.Module):
    """Base encoder followed by spatial squeeze to produce latent vectors.

    Keeps the four downsampling stages but collapses spatial support with
    adaptive average pooling so the decoder receives true bottleneck codes.
    """
    def __init__(self, initial_filters=4, latent_dim=128, bottleneck_shape=(1,1,1)):
        super().__init__()
        self.backbone = BaseEncoder(initial_filters=initial_filters)
        # Conv3d projects from 128 channels to latent_dim while collapsing spatial dims
        self.global_pool = nn.Conv3d(128, latent_dim, kernel_size=(4,8,8), padding=0)
        self.latent_dim = latent_dim

    def forward(self, x):
        # Extract feature volume using the shared backbone.
        x = self.backbone(x)
        # Collapse spatial dimensions and project to latent_dim
        x = self.global_pool(x)  # -> (B, latent_dim, 1, 1, 1)
        return x


class BaseDecoder(nn.Module):
    """Symmetric decoder that upsamples tensors of shape (latent_dim, 4, 8, 8) back to (output_channels, 64, 128, 128).
    The decoder mirrors the encoder's depth, using trilinear upsampling followed by feature mixing via convolutional blocks."""
    def __init__(self, latent_dim=128, output_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Upsampling path mirrors encoder depth with trilinear resize then feature mixing
        # Input tensor: (latent_dim, 4, 8, 8)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # -> (latent_dim, 8, 16, 16)
            BaseConvBlock(latent_dim, 64), # -> (64, 8, 16, 16)
        )

        # Input tensor: (64, 8, 16, 16)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # -> (64, 16, 32, 32)
            BaseConvBlock(64, 32), # -> (32, 16, 32, 32)
        )

        # Input tensor: (32, 16, 32, 32)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # -> (32, 32, 64, 64)
            BaseConvBlock(32, 16), # -> (16, 32, 64, 64)
        )

        # Input tensor: (16, 32, 64, 64)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # -> (16, 64, 128, 128)
            BaseConvBlock(16, 4), # -> (4, 64, 128, 128)
        )

        # Final convolution to produce the output image
        self.final_conv = nn.Conv3d(4, output_channels, kernel_size=1) # -> (output_channels, 64, 128, 128)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        return x


class BottleneckBasisDecoder(nn.Module):
    """Your basis decoder: upsample latent vector to initial spatial size, depthwise conv (kernel==spatial),
       1x1 channel mixing, then ladder."""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=None,
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
        groups='full',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        stages = 4
        if initial_shape is None:
            self.initial_shape = tuple(max(1, s // (2 ** stages)) for s in target_shape)
        else:
            self.initial_shape = initial_shape
        self.reshape = nn.Upsample(size=self.initial_shape, mode='trilinear', align_corners=False) # we should use the nearest model
        if groups == 'full':
            group_count = latent_dim
        elif groups == 'none':
            group_count = 1
        elif groups == 'partial':
            # Use max(1, latent_dim // 4) to mimic notebook variant counts (e.g., 32→8, 64→16).
            group_count = max(1, latent_dim // 4)
        elif isinstance(groups, int):
            group_count = groups
        else:
            raise ValueError(f"Unsupported groups spec: {groups}")
        self.groups = groups
        # depthwise kernel spans the full initial_shape -> each latent maps to one basis patch
        self.grouped_conv = nn.Conv3d(
            latent_dim,
            latent_dim,
            kernel_size=self.initial_shape,
            groups=group_count,
            padding='same',
            bias=True,
        )
        # Mixer only needed when groups != 'none'
        self.mixer = nn.Conv3d(latent_dim, mid_channels, kernel_size=1) if groups != 'none' else None
        self.act = nn.ReLU(inplace=True)
        
        # Adaptive ladder: when groups=='none', first stage accepts latent_dim; otherwise mid_channels
        ladder_shapes = _compute_ladder_shapes(self.initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        
        # First ladder stage input channels depend on whether mixer is used
        in_channels = latent_dim if groups == 'none' else mid_channels
        current_shape = self.initial_shape
        
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            current_shape = target
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, latent_dim, bz, by, bx) -- e.g., (B, latent_dim, 1,1,1)
        # b = x.size(0)
        # x = x.view(b, self.latent_dim, x.shape[2], x.shape[3], x.shape[4]) # why do we need those 2 lines!
        x = self.reshape(x)         # -> (B, latent_dim, D,H,W) with D,H,W == initial_shape
        # grouped_conv uses kernel == current spatial size; ensure kernel and input shape match
        x = self.grouped_conv(x)
        if self.mixer is not None:
            x = self.mixer(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


class BottleneckDeconvDecoder(nn.Module):
    """ConvTranspose3d variant: densly learn spatial expansion from the latent vector."""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=None,
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.target_shape = target_shape
        stages = 4
        if initial_shape is None:
            self.initial_shape = tuple(max(1, s // (2 ** stages)) for s in target_shape)
        else:
            self.initial_shape = initial_shape
        self.deconv = nn.ConvTranspose3d(latent_dim, mid_channels, kernel_size=self.initial_shape)
        self.act = nn.ReLU(inplace=True)
        ladder_shapes = _compute_ladder_shapes(self.initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        current_shape = self.initial_shape
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            current_shape = target
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    def forward(self, x):
        # b = x.size(0)
        # x = x.view(b, x.size(1), x.shape[2], x.shape[3], x.shape[4])  # usually (B, latent_dim, 1,1,1)
        x = self.deconv(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


class ComplexBottleneckDeconvDecoder(nn.Module):
    """ConvTranspose3d variant: densly learn spatial expansion from the latent vector."""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=None,
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.target_shape = target_shape
        stages = 4
        if initial_shape is None:
            self.initial_shape = tuple(max(1, s // (2 ** stages)) for s in target_shape)
        else:
            self.initial_shape = initial_shape
        self.deconv = nn.ConvTranspose3d(latent_dim, mid_channels, kernel_size=self.initial_shape)
        self.act = nn.ReLU(inplace=True)
        ladder_shapes = _compute_ladder_shapes(self.initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        current_shape = self.initial_shape
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            current_shape = target
        self.refine_conv = BaseConvBlock(in_channels, in_channels)
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    def forward(self, x):
        # b = x.size(0)
        # x = x.view(b, x.size(1), x.shape[2], x.shape[3], x.shape[4])  # usually (B, latent_dim, 1,1,1)
        x = self.deconv(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        x = self.refine_conv(x)
        return self.final_conv(x)


class BottleneckHybridDecoder(nn.Module):
    """Hybrid: depthwise with small kernel -> mixer -> small ConvTranspose -> ladder."""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=None,
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        stages = 4
        if initial_shape is None:
            self.initial_shape = tuple(max(1, s // (2 ** stages)) for s in target_shape)
        else:
            self.initial_shape = initial_shape
        self.reshape = nn.Upsample(size=self.initial_shape, mode='trilinear', align_corners=False)
        # small depthwise kernel to capture local patterns
        self.grouped_conv = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=latent_dim)
        self.mixer = nn.Conv3d(latent_dim, mid_channels, kernel_size=1)
        # small deconv to add learned spatial detail
        self.small_deconv = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=2, stride=1)
        self.act = nn.ReLU(inplace=True)
        # ladder
        ladder_shapes = _compute_ladder_shapes(self.initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        current_shape = self.initial_shape
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            current_shape = target
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, self.latent_dim, x.shape[2], x.shape[3], x.shape[4])
        x = self.reshape(x)
        x = self.grouped_conv(x)
        x = self.mixer(x)
        x = self.small_deconv(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


class BaseModel(nn.Module):
    """Wrapper that wires the base encoder and decoder into a full 3D convolutional autoencoder."""
    def __init__(self, initial_filters=4, latent_dim=128, use_bottleneck=False):
        super().__init__()
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            # Latent vector pathway: compress spatial dims and decode via grouped bottleneck.
            self.encoder = BottleneckEncoder(initial_filters=initial_filters, latent_dim=latent_dim)
            self.decoder = BottleneckBasisDecoder(latent_dim=latent_dim)
        else:
            # Original autoencoder pathway: keep spatial latent map for the vanilla decoder.
            self.encoder = BaseEncoder(initial_filters=initial_filters)
            self.decoder = BaseDecoder(latent_dim=latent_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


# Wrapper that expects encoder output to be spatial (e.g., BaseEncoder -> BaseDecoder)
class SpatialAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

# Wrapper for bottleneck-style encoder that outputs (B, latent_dim, bz,by,bx) and dec expects that
class BottleneckAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class BottleneckBasisDecoderFixed(nn.Module):
    """Fixed basis decoder with reasonable kernel sizes"""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=(4, 8, 8),
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
        groups='full',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_shape = initial_shape
        self.target_shape = target_shape
        
        # Upsample from bottleneck to initial_shape
        self.reshape = nn.Upsample(size=initial_shape, mode='trilinear', align_corners=False)
        
        # Group settings
        if groups == 'full':
            group_count = latent_dim
        elif groups == 'none':
            group_count = 1
        elif groups == 'partial':
            group_count = max(1, latent_dim // 4)
        elif isinstance(groups, int):
            group_count = groups
        else:
            raise ValueError(f"Unsupported groups spec: {groups}")
        
        # FIX: Use 3x3x3 depthwise conv instead of full spatial kernel
        self.grouped_conv = nn.Conv3d(
            latent_dim,
            latent_dim,
            kernel_size=3,  # Fixed: reasonable kernel size
            groups=group_count,
            padding=1,
            bias=True,
        )
        
        # Add batch norm and activation after grouped conv
        self.bn = nn.BatchNorm3d(latent_dim)
        
        # Channel mixer
        self.mixer = nn.Conv3d(latent_dim, mid_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        
        # Decoder ladder
        ladder_shapes = _compute_ladder_shapes(initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
        
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, self.latent_dim, x.shape[2], x.shape[3], x.shape[4])
        x = self.reshape(x)
        x = self.grouped_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.mixer(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


class BottleneckDeconvDecoderFixed(nn.Module):
    """Fixed deconv decoder with progressive upsampling"""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=(4, 8, 8),
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.initial_shape = initial_shape
        self.target_shape = target_shape
        
        # FIX: Use smaller deconv with stride instead of huge kernel
        self.deconv1 = nn.ConvTranspose3d(latent_dim, mid_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        # Additional conv to refine
        self.refine = BaseConvBlock(mid_channels, mid_channels)
        
        # Decoder ladder
        ladder_shapes = _compute_ladder_shapes(initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
        
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, x.size(1), x.shape[2], x.shape[3], x.shape[4])
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.refine(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


class BottleneckHybridDecoderFixed(nn.Module):
    """Fixed hybrid decoder with proper normalization"""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=(4, 8, 8),
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_shape = initial_shape
        self.target_shape = target_shape
        self.reshape = nn.Upsample(size=initial_shape, mode='trilinear', align_corners=False)
        
        # Small depthwise kernel
        self.grouped_conv = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=latent_dim)
        self.bn1 = nn.BatchNorm3d(latent_dim)  # ✓ ADD THIS
        self.act1 = nn.ReLU(inplace=True)
        
        # Channel mixer
        self.mixer = nn.Conv3d(latent_dim, mid_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(mid_channels)  # ✓ ADD THIS
        self.act2 = nn.ReLU(inplace=True)
        
        # Small deconv
        self.small_deconv = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm3d(mid_channels)  # ✓ ADD THIS
        self.act3 = nn.ReLU(inplace=True)
        
        # Ladder
        ladder_shapes = _compute_ladder_shapes(initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
        
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, self.latent_dim, x.shape[2], x.shape[3], x.shape[4])
        x = self.reshape(x)
        x = self.grouped_conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.mixer(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.small_deconv(x)
        x = self.bn3(x)
        x = self.act3(x)
        for stage in self.ladder:
            x = stage(x)
        return self.final_conv(x)


