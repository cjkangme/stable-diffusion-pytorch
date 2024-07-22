import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, d_embed: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, d_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, d_embed))

    # nn.Embedding으로 초기화된 임베딩 및 positional embedding은 모두 훈련 중 학습됨
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # (B, seq_len) -> (B, seq_len, d_embed)
        x = self.token_embedding(tokens)
        # Add Positional Embedding
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, d_embed: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_head, d_embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_embed)

        residual = x
        x = self.layernorm_1(x)
        x = self.attention(
            x, casual_mask=True
        )  # 언어 임베딩이므로 미래의 입력을 받지 않는 마스킹 필요
        x += residual

        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # GELU activation function의 빠른 계산법
        x = self.linear_2(x)
        x += residual

        # x: (B, seq_len, d_embed)
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # 49408개의 단어로 구성된 vocab 사용
        self.embedding = CLIPEmbedding(49408, 768, 77)
        # 12는 multi-head-attention
        self.layers = nn.Module([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)

    # vocab의 인덱스가 들어오기 때문에 LongTensor 사용
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (B, seq_len) -> (B, seq_len, d_embed)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (B, seq_len, d_embed)
        output = self.layernorm(state)

        return output
