import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .linear_attention import LinearAttnFFN
from .base_layers import LayerNorm
from .masking import random_masking


class SemanticMaskModule(nn.Module):
    """LiMR semantic masked reconstruction module.

    Inserted between frozen stage1 output and trainable stage2 input (~N=2.5).
    Corresponds to f_LV(f_Ex(x_n) | M_3, I_3) in paper Eq.1.

    Training: x*mask -> local_rep -> unfold -> gather(visible) -> LinearAttnFFN -> fold -> residual
    Inference: lightweight pass-through with same residual structure
    """

    def __init__(
        self,
        in_channels: int,
        mask_ratio: float = 0.4,
        attn_dim: int | None = None,
        num_attn_blocks: int = 2,
        patch_size: int = 2,
        mask_grid_size: int = 7,
        dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.mask_grid_size = mask_grid_size
        self.in_channels = in_channels
        attn_dim = attn_dim or (in_channels // 2)

        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels, attn_dim, 1, bias=False),
        )

        blocks = []
        for _ in range(num_attn_blocks):
            blocks.append(
                LinearAttnFFN(
                    embed_dim=attn_dim,
                    ffn_latent_dim=attn_dim * 2,
                    attn_dropout=0.0,
                    dropout=dropout,
                    ffn_dropout=ffn_dropout,
                )
            )
        blocks.append(LayerNorm(attn_dim))
        self.transformer = nn.Sequential(*blocks)

        self.proj = nn.Sequential(
            nn.Conv2d(attn_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_ratio == 0.0:
            return self._inference_forward(x)
        return self._masked_forward(x)

    def _masked_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        PH, PW = self.patch_size, self.patch_size

        mask, idx_keep, ids_restore = self._make_mask(x)

        local = self.local_rep(x * mask)

        patches = F.unfold(local, kernel_size=(PH, PW), stride=(PH, PW))
        patches = patches.reshape(B, -1, PH * PW, (H // PH) * (W // PW))

        patches = torch.gather(
            patches, 3,
            idx_keep.view(B, 1, 1, -1).expand(-1, patches.shape[1], PH * PW, -1)
        )

        patches = self.transformer(patches)

        N_total = (H // PH) * (W // PW)
        n_masked = N_total - patches.shape[3]
        if n_masked > 0:
            zeros = torch.zeros(B, patches.shape[1], PH * PW, n_masked,
                                device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, zeros], dim=3)
        patches = torch.gather(
            patches, 3,
            ids_restore.view(B, 1, 1, -1).expand(-1, patches.shape[1], PH * PW, -1)
        )

        fm = F.fold(
            patches.reshape(B, -1, N_total),
            output_size=(H, W),
            kernel_size=(PH, PW),
            stride=(PH, PW),
        )

        # return x + self.proj(fm)
        return self.proj(fm)

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        PH, PW = self.patch_size, self.patch_size

        local = self.local_rep(x)

        patches = F.unfold(local, kernel_size=(PH, PW), stride=(PH, PW))
        patches = patches.reshape(B, -1, PH * PW, (H // PH) * (W // PW))

        patches = self.transformer(patches)

        patches = patches.reshape(B, -1, (H // PH) * (W // PW))
        fm = F.fold(patches, output_size=(H, W), kernel_size=(PH, PW), stride=(PH, PW))

        return self.proj(fm)

    def _make_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape

        _, mask_coarse, _ = random_masking(x, self.mask_ratio, grid_size=self.mask_grid_size)
        gs = self.mask_grid_size
        mask_coarse = mask_coarse.reshape(B, 1, gs, gs)

        mask_spatial = F.interpolate(mask_coarse.float(), size=(H, W), mode='nearest')

        mask_binary = (1 - mask_spatial).float()

        mask_for_patches = F.unfold(
            mask_binary,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )
        mask_for_patches = mask_for_patches.reshape(B, 1, self.patch_size * self.patch_size, -1)
        idx_keep = mask_for_patches[0, 0, 0, :].nonzero(as_tuple=True)[0]

        idx_mask = (1 - mask_for_patches[0, 0, 0, :]).nonzero(as_tuple=True)[0]
        ids_shuffle = torch.cat([idx_keep, idx_mask], dim=0)
        ids_restore_local = torch.argsort(ids_shuffle)
        idx_keep = idx_keep.unsqueeze(0).repeat(B, 1)
        ids_restore_local = ids_restore_local.unsqueeze(0).repeat(B, 1)

        return mask_binary, idx_keep, ids_restore_local
