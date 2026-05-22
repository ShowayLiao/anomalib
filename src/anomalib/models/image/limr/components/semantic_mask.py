import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .linear_attention import LinearAttnFFN
from .base_layers import LayerNorm


class SemanticMaskModule(nn.Module):
    """LiMR 语义掩码重建模块 (完全批处理优化版)
    
    去除了最后的残差连接，并完全修复了多 Batch 训练时索引不匹配的致命 Bug。
    """

    def __init__(
        self,
        in_channels: int,
        mask_ratio: float = 0.4,
        attn_dim: int | None = None,
        num_attn_blocks: int = 2,
        patch_size: int = 2,
        mask_grid_size: int = 8,  # 🚀 关键修改：从 7 改为 8，完美整除 32 (64//2)
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
        HP, WP = H // PH, W // PW
        N_total = HP * WP

        # 1. 获得全 Batch 齐整的掩码及索引
        mask_binary, idx_keep, ids_restore = self._make_mask_batched(x)

        # 2. 空间掩码
        local = self.local_rep(x * mask_binary)

        # 3. 切分 Patch 并展平
        patches = F.unfold(local, kernel_size=(PH, PW), stride=(PH, PW))
        patches = patches.reshape(B, -1, PH * PW, N_total)

        # 4. 完美的 Batched Gather (保留可见 Patch)
        num_keep = idx_keep.shape[1]
        idx_keep_expanded = idx_keep.view(B, 1, 1, num_keep).expand(-1, patches.shape[1], PH * PW, -1)
        patches = torch.gather(patches, 3, idx_keep_expanded)

        # 5. Transformer 提取上下文特征
        patches = self.transformer(patches)

        # 6. 补零并还原空间位置
        n_masked = N_total - num_keep
        if n_masked > 0:
            zeros = torch.zeros(B, patches.shape[1], PH * PW, n_masked,
                                device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, zeros], dim=3)
        
        ids_restore_expanded = ids_restore.view(B, 1, 1, N_total).expand(-1, patches.shape[1], PH * PW, -1)
        patches = torch.gather(patches, 3, ids_restore_expanded)

        # 7. Fold 回成二维特征图
        fm = F.fold(
            patches.reshape(B, -1, N_total),
            output_size=(H, W),
            kernel_size=(PH, PW),
            stride=(PH, PW),
        )

        # 🚀 遵循纯粹的无残差自监督重建瓶颈
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

    def _make_mask_batched(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        PH, PW = self.patch_size, self.patch_size
        HP, WP = H // PH, W // PW
        GS = self.mask_grid_size

        # 每个粗糙掩码格子内部包含的 Patch 数量 (由于 32 被 8 整除，这里刚好是 4)
        BH, BW = HP // GS, WP // GS

        # 1. 在粗糙网格尺度上生成独立的随机噪声并排序
        noise_coarse = torch.rand(B, GS * GS, device=x.device)
        ids_shuffle_coarse = torch.argsort(noise_coarse, dim=1)
        ids_restore_coarse = torch.argsort(ids_shuffle_coarse, dim=1)

        # 2. 决定保留哪些粗糙网格 (1为可见，0为掩码)
        num_keep_coarse = int(GS * GS * (1 - self.mask_ratio))
        mask_coarse = torch.zeros(B, GS * GS, device=x.device)
        mask_coarse[:, :num_keep_coarse] = 1
        mask_coarse = torch.gather(mask_coarse, 1, ids_restore_coarse).reshape(B, GS, GS)

        # 3. 将网格掩码无损放大到 Patch 尺度 [B, HP, WP]
        mask_patches = torch.repeat_interleave(
            torch.repeat_interleave(mask_coarse, BH, dim=1), BW, dim=2
        )
        mask_patches_flat = mask_patches.flatten(1)  # [B, HP * WP]

        # 4. 利用降序排列，把所有可见的 1 顶到最前面，被掩码的 0 排在后面
        # 这样可以天然确保每个 Batch 元素的 Keep 索引长度完全整齐一致！
        ids_shuffle = torch.argsort(mask_patches_flat, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 5. 截取统一长度的 idx_keep
        num_keep_patches = num_keep_coarse * BH * BW
        idx_keep = ids_shuffle[:, :num_keep_patches]

        # 6. 将 Patch 尺度的掩码放大到像素级尺度 [B, 1, H, W] 用于对输入做空间掩码
        mask_spatial = torch.repeat_interleave(
            torch.repeat_interleave(mask_patches.unsqueeze(1), PH, dim=2), PW, dim=3
        )

        return mask_spatial, idx_keep, ids_restore