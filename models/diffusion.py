import torch
from torch import nn
from torch.nn import functional as F
from models.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        x = F.silu()
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x


class UNET_ResidualBlock(nn.module):
    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(d_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature, time):
        # feature: (B, in_channels, H, W)
        # time: (1, 1280)
        residual = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        # (1, 1280) -> (1, out_channels)
        time = self.linear_time(time)
        # (1, out_channels) -> (1, out_channels, 1, 1) for broadcasting
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context: int = 768):
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        # LayerNormalization으로 이미 정규화 되어있기 때문에 bias가 불필요한 오버헤드를 초래할 수 있다고 함
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads, channels, d_context, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (B, C, H, W)
        # context: (B, seq_len, dim)

        residual_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (B, C, H, W) -> (B, C, H*W)
        x = x.view((n, c, h * w))
        # (B, C, H*W) -> (B, H*W, C)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residual_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residual_short

        # Normalization + Cross Attention with skip connection
        residual_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residual_short

        # Normalization + Feed Foward with GeGLU and skip connection
        residual_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x, gate = self.linear_geglu_2(x)
        x += residual_short

        # (B, H*W, C) -> (B, C, H*W)
        x = x.transpose(-1, -2)
        # (B, C, H*W) -> (B, C, H, W)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residual_long


class Upsample(nn.module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, H * 2, W * 2)
        x = F.interpolate(
            x, scale_factor=2, mode="nearest"
        )  # 기존의 nn.Upsample과 동일한 로직
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)  # cross attention
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.Module(
            [
                # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # (B, 320, H / 8, W / 8) -> (B, 320, H / 16, W / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
                ),
                # (B, 640, H / 16, W / 16) -> (B, 640, H / 32, W / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (B, 1280, H / 32, W / 32) -> (B, 1280, H / 64, W / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                # (B, 1280, H / 64, W / 64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.Module(
            [
                # Bottleneck input이 skip connection을 통해 concat 되므로 채널이 2배가 됨
                # (B, 2560, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 32, W / 32)
                SwitchSequential(UNET_ResidualBlock(1280, 1280), Upsample(1280)),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (B, 1280, H / 32, W / 32) -> (B, 1280, H / 16, W / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                # (B, 640, H / 16, W / 16) -> (B, 640, H / 8, W / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)
                ),  # TODO: 이건 뭐지?
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
            ]
        )


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 320, H / 8, W / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # x: (B, 3, H / 8, W / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # timestep는 320 크기의 임베딩으로 들어감. (모델이 이해가능한 형태로 변환)
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent: (B, 4, H / 8, W / 8)
        # context: (B, seq_len, d_embed)
        # time_embed: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
        output = self.unet(latent, context, time)
        # (B, 320, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
        output = self.final(output)

        return output
