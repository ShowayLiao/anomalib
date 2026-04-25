# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transformer blocks for INP-Former model.

This module implements the Transformer blocks used in INP-Former:
1. Prototype_Block: Used in the decoder with Prototype_Attention
2. Aggregation_Block: Used in the INP Extractor with Aggregation_Attention
"""

from torch import nn
from .attention import Aggregation_Attention, Prototype_Attention
from .mlp import Mlp


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Prototype_Block(nn.Module):
    """Transformer block with Prototype Attention for INP-Guided Decoder.

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for attention scores. Defaults to None.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        drop_path (float, optional): Drop path probability. Defaults to 0.0.
        act_layer (callable, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (callable, optional): Normalization layer. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: callable = nn.GELU,
        norm_layer: callable = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Prototype_Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, prototype, return_attention=False):
        """Forward pass through Prototype Block.

        Args:
            x (torch.Tensor): Input features.
            prototype (torch.Tensor): INP prototypes.
            return_attention (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Output features and optionally attention weights.
        """
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        # Remove first residual connection to prevent anomaly information from propagating
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class Aggregation_Block(nn.Module):
    """Transformer block with Aggregation Attention for INP Extractor.

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for attention scores. Defaults to None.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        drop_path (float, optional): Drop path probability. Defaults to 0.0.
        act_layer (callable, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (callable, optional): Normalization layer. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: callable = nn.GELU,
        norm_layer: callable = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Aggregation_Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, y):
        """Forward pass through Aggregation Block.

        Args:
            x (torch.Tensor): Learnable tokens (queries).
            y (torch.Tensor): Image features (keys and values).

        Returns:
            torch.Tensor: Aggregated INP prototypes.
        """
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x