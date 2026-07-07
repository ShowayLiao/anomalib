import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Sequence

from .linear_attention import LinearAttnFFN
from .base_layers import LayerNorm


class MobileViTBlockv2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.4,
        ffn_dropout: Optional[float] = 0.1,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                dilation=dilation,
                groups=in_channels,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

        conv_1x1_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        super(MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=cnn_out_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels),
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim

    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
    ) -> Tuple[nn.Module, int]:
        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(LayerNorm(normalized_shape=d_model))

        return nn.Sequential(*global_rep), d_model

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.resize_input_if_needed(x)
        fm_conv = self.local_rep(x)

        patches = F.unfold(
            fm_conv,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            x.shape[0], self.cnn_out_dim, self.patch_h * self.patch_w, -1
        )

        patches = self.global_rep(patches)

        patches = patches.reshape(x.shape[0], -1, patches.shape[3])
        fm = F.fold(
            patches,
            output_size=(x.shape[2], x.shape[3]),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        output = self.conv_proj(fm)
        return output

    def unfolding_pytorch(self, feature_map, mask=None, idx_keep=None):
        batch_size, in_channels, img_h, img_w = feature_map.shape

        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        if mask is not None:
            patches = patches.reshape(
                batch_size, in_channels, self.patch_h * self.patch_w, -1
            )
            feature_patches = torch.gather(patches, 3,
                idx_keep.view([feature_map.shape[0], 1, 1, -1]).
                expand(batch_size, in_channels, self.patch_h * self.patch_w, -1))
        else:
            patches = patches.reshape(
                batch_size, in_channels, self.patch_h * self.patch_w, -1
            )
            feature_patches = patches

        return feature_patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size, ids_restore=None):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        if ids_restore is not None:
            mask_token = torch.zeros(
                (batch_size, in_dim * patch_size, ids_restore.shape[1] - n_patches),
                device=patches.device
            )
            patches = torch.cat([patches, mask_token], dim=2)
            patches = torch.gather(patches, 2,
                ids_restore.view([batch_size, 1, -1]).
                expand(batch_size, in_dim * patch_size, -1))

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward_masked(self, x, mask=None, idx_keep=None, ids_restore=None):
        x = self.resize_input_if_needed(x)

        if mask is not None:
            x = x * mask
        fm_conv = self.local_rep(x)

        patches, output_size = self.unfolding_pytorch(fm_conv, mask, idx_keep)

        patches = self.global_rep(patches)

        fm = self.folding_pytorch(patches, output_size, ids_restore)

        output = self.conv_proj(fm)
        return output
