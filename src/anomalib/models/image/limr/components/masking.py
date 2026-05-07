import numpy as np
import torch
from torch.nn import functional as F


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def random_masking(x, mask_ratio):
    N = x.shape[0]
    H, W = x.shape[2], x.shape[3]
    L = (H // 32) * (W // 32)
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return ids_keep, mask, ids_restore


def mask_upsample(mask, H_low, H_high, batch_size):
    up_mask = (
        1 - mask.reshape(-1, H_high, H_high)
        .unsqueeze(-1)
        .repeat(1, 1, 1, H_low ** 2 // H_high ** 2)
        .reshape(-1, H_high, H_high, H_low // H_high, H_low // H_high)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size, H_low, H_low)
        .unsqueeze(1)
    )
    return up_mask


def mask2ids(mask, path_size=2):
    mask = F.unfold(mask, kernel_size=(path_size, path_size), stride=(path_size, path_size))
    mask = mask.reshape(mask.shape[0], 1, path_size * path_size, -1)
    idx_keep = mask[0, 0, 0, :].nonzero(as_tuple=True)[0]
    idx_mask = (1 - mask[0, 0, 0, :]).nonzero(as_tuple=True)[0]
    ids_shuffle = torch.cat([idx_keep, idx_mask], dim=0)
    ids_restore = torch.argsort(ids_shuffle)
    return idx_keep, ids_restore
