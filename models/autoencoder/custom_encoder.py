import torch
import sys
sys.path.append('../../models')

import torch
import torch.nn as nn
from autoencoder.selected_model import BaseConvBlock


CHECKPOINT_PATH = '../../models/checkpoints/beta_5.00e-05_best.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomEncoder(nn.Module):
    def __init__(self, initial_filters=32, latent_dim=256):
        super().__init__()
        f = initial_filters
        
        # Use 'backbone' to match checkpoint key structure
        self.backbone = nn.Module()
        self.backbone.down1 = nn.Sequential(
            BaseConvBlock(1,   f,   stride=2),
            BaseConvBlock(f,   f)
        )
        self.backbone.down2 = nn.Sequential(
            BaseConvBlock(f,   f*2, stride=2),
            BaseConvBlock(f*2, f*2)
        )
        self.backbone.down3 = nn.Sequential(
            BaseConvBlock(f*2, f*4, stride=2),
            BaseConvBlock(f*4, f*4)
        )
        self.backbone.down4 = nn.Sequential(
            BaseConvBlock(f*4, f*8, stride=2),
            BaseConvBlock(f*8, f*8)
        )
        self.global_pool = nn.Conv3d(
            f*8, latent_dim * 2,
            kernel_size=(4, 8, 8), padding=0
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.backbone.down1(x)
        x = self.backbone.down2(x)
        x = self.backbone.down3(x)
        x = self.backbone.down4(x)
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        mu      = x[:, :self.latent_dim]
        log_var = x[:, self.latent_dim:]
        return mu, log_var