import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from torch.nn import Conv2d, Dropout
import torch.nn as nn

from .base_layers import LayerNorm


class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.qkv_proj = Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            padding=0,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.bias[: self.embed_dim + 1, ...] if self.qkv_proj.bias is not None else None,
        )
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        value = F.conv2d(
            x,
            weight=self.qkv_proj.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.bias[self.embed_dim + 1 :, ...] if self.qkv_proj.bias is not None else None,
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)


class LinearAttnFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            LayerNorm(normalized_shape=embed_dim),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            LayerNorm(normalized_shape=embed_dim),
            Conv2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                padding=0,
            ),
            nn.LeakyReLU(negative_slope=0.1),
            Dropout(p=ffn_dropout),
            Conv2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                padding=0,
            ),
            Dropout(p=dropout),
        )

    def forward(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
        if x_prev is None:
            x = x + self.pre_norm_attn(x)
        else:
            res = x
            x = self.pre_norm_attn[0](x)
            x = self.pre_norm_attn[1](x, x_prev)
            x = self.pre_norm_attn[2](x)
            x = x + res

        x = x + self.pre_norm_ffn(x)
        return x
