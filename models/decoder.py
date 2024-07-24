import torch
from torch import nn
from torch.nn import functional as F
from models.attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, channels, H, W)

        residual = x

        n, c, h, w = x.shape
        # (B, channels, H, W) -> (B, channels, H*W)
        x = x.view((n, c, h * w))
        # 행렬 연산을 위한 Transpose
        # (B channels, H*W) -> (B, H*W, channels)
        x = x.transpose(-1, -2)
        x = self.attention(x)
        # (B, H*W, channels) -> (B, channels, H*W)
        x = self.transpose(-1, -2)
        # (B, channels, H*W) -> (B, channels, H, W)
        x = self.view((n, c, h, w))
        x = self.groupnorm(x)

        return x + residual


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_chennels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_chennels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_chennels)
        self.conv_2 = nn.Conv2d(out_chennels, out_chennels, kernel_size=3, padding=1)

        if in_channels == out_chennels:
            self.residual_layer = nn.Identity()
        else:
            # skip connection 수행 시 채널 수를 맞추기 위한 1x1 conv 수행
            self.residual_layer = nn.Conv2d(
                in_channels, out_chennels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)

        # skip connection을 위해  x 저장
        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        # (B, out_channels, H, W) -> (B, out_channels, H, W)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Upsample(512, scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/4, W/4) -> (B, 512, H/2, W/2)
            nn.Upsample(512, scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (B, 256, H/2, W/2) -> (B, 256, H, W)
            nn.Upsample(256, scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (B, 128, H, W) -> (B, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H/8, W/8)
        # Encoder에서 했던 scaling 반대로
        x /= 0.18215

        for module in self:
            x = module(x)
        # x: (B, 3, H, W)
        return x
