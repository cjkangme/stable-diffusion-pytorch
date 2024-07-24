import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (B, seq_len, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        intermid_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (B, seq_len, dim) -> (B, seq_len, dim * 3) -> 3 tensors
        query, key, value = torch.chunk(self.in_proj(x), 3, dim=-1)
        # (B, seq_len, dim) -> (B, H, sequence_length, dim / H)
        query = query.view(intermid_shape).transpose(1, 2)
        key = key.view(intermid_shape).transpose(1, 2)
        value = value.view(intermid_shape).transpose(1, 2)
        # (B, H, sequence_length, dim / H) @ (B, H, dim / H, sequence_length)
        # = (B, H, sequence_length, sequence_length)
        weight = query @ key.transpose(-1, -2)

        if casual_mask:
            # tensor.triu는 행렬을 0으로 마스킹하여 upper triangle 행렬을 만듦
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # bool mask가 true인 부분을 두번째 인자 값으로 채움
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (B, H, sequence_length, sequence_length) @ (B, H, sequence_length, dim / H)
        # = (B, H, sequence_length, dim / H)
        output = weight @ value
        # (B, H, sequence_length, dim / H) -> (B, sequence_length, dim)
        output = output.transpose(1, 2).view(input_shape)

        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads:int, d_embed:int, d_cross:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # 동일한 matrix 3개를 만들면 되는 SelfAttention과 달리 2종류의 Matrix를 만들어야 함
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, context:torch.Tensor):
        # x: (latent): (B, seq_len_q, dim_q)
        # context: (embed): (B, seq_len_kv, dim_kv) = (Batch_Size, 77, 768)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # seq_len이 다를 수 있으므로 고정값이 아닌  -1로 처리
        intermid_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (B, seq_len_q, dim_q)
        query = self.q_proj(x)
        # (B, seq_len_kv, dim_q)
        key = self.k_proj(context)
        value = self.v_proj(context)

        # (B, n_heads, seq_len_q, d_head)
        query = query.view(intermid_shape).transpose(1, 2)
        # (B, n_heads, seq_len_kv, d_head)
        key = key.view(intermid_shape).transpose(1, 2)
        value = value.view(intermid_shape).transpose(1, 2)

        # (B, n_heads, seq_len_q, seq_len_kv)
        weight = query @ key.transpose(-1, -2)
        # 마스크가 필요 없으므로 곧바로 연산
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, n_heads, seq_len_q, d_head)
        output = weight @ value
        # (B, seq_len_q, n_heads, d_head)
        output = output.transpose(1, 2).continuous() # 메모리를 새롭게 할당하여 연속적으로 저장되도록 만듦
        # (B, seq_len_q, dim_q)
        output = output.view(input_shape)
        output = self.out_proj(output)
        # (B, seq_len_q, dim_q)
        return output

