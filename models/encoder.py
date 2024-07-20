import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlcok, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # 이 모델의 구조는 경험적으로 개선되어온 구조임
        super().__init__(
            # (B, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) -> (B, 128, H / 2, W / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (B, 128, H / 2, W / 2) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(128, 256),
            # (B, 256, H / 2, W / 2) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(256, 256),
            # (B, 256, H / 2, W / 2) -> (B, 256, H / 4, W / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (B, 256, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(256, 512),
            # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 4, W / 4) -> (B, 512, H / 8, W / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlcok(512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(
                32, 512
            ),  # Batch 사이즈가 극도로 작은 상황에서 Batch Normailization 대신 사용하면 좋은 결과를 얻을 수 있다 함
            nn.SiLU(),
            # (B, 512, H / 8, W / 8) -> (B, 8, H / 8, W / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (B, 8, H / 8, W / 8) -> (B, 8, H / 8, W / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # noise: (B, out_channels, H / 8, W / 8)

        for module in self:
            # stride가 들어간 layer일 경우 output을 input_size/2로 맞추기 위해 asymmetric padding
            if getattr(module, "stride", None) == (2, 2):
                # (left, right, top, bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (B, 8, H / 8, W / 8) -> (B, 4, H / 8, W / 8) * 2
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # 최소가 -30, 최대가 20이 되도록 클립핑
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # VAE가 샘플링하는 법
        # Z = N(0, 1) -> X = N(mean, variance)
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # Scaling, 스케일 상수는 논문에서 제시된 수치 사용
        x *= 0.18215

        return x
