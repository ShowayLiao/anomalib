import logging

import torch
from torch import nn
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

from .base_layers import InvertedResidual
from .masking import random_masking
from .mobilevit_v2_block import MobileViTBlockv2 as Block


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(min_val, max_val, value):
    return max(min_val, min(max_val, value))


class MobileViTv2(nn.Module):
    def __init__(self, width_multiplier, block_ffn_dropout=0.1, block_attn_dropout=0.0, block_dropout=0.1) -> None:
        super().__init__()

        self.dilation = 1
        self.dilate_l4 = False
        self.dilate_l5 = False

        ffn_multiplier = 2
        mv2_exp_mult = 2

        layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
        layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))

        mobilevit_config = {
            "layer0": {
                "img_channels": 3,
                "out_channels": layer_0_dim,
            },
            "layer1": {
                "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 2,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {
                "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
                "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
                "ffn_multiplier": ffn_multiplier,
                "attn_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "block_type": "mobilevit",
            },
            "layer4": {
                "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
                "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
                "ffn_multiplier": ffn_multiplier,
                "attn_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "block_type": "mobilevit",
            },
            "layer5": {
                "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
                "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
                "ffn_multiplier": ffn_multiplier,
                "attn_blocks": 3,
                "patch_h": 1,
                "patch_w": 1,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer3"],
            block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
        )

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
            block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
        )

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
            block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
        )

    def _make_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False,
        block_ffn_dropout=0.1, block_attn_dropout=0.0, block_dropout=0.1,
    ) -> Tuple[nn.ModuleList, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate,
                block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
                block_dropout=block_dropout,
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg
            )

    def _make_mobilenet_layer(self, input_channel: int, cfg: Dict) -> Tuple[nn.ModuleList, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = nn.ModuleList()

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )

            block.append(layer)
            input_channel = output_channels

        return block, input_channel

    def _make_mit_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False,
        block_ffn_dropout=0.1, block_attn_dropout=0.0, block_dropout=0.1,
    ) -> Tuple[nn.ModuleList, int]:
        prev_dilation = self.dilation
        block = nn.ModuleList()
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = block_dropout

        block.append(
            Block(
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=block_ffn_dropout,
                attn_dropout=block_attn_dropout,
                conv_ksize=3,
                dilation=self.dilation,
            )
        )

        return block, input_channel

    def forward(self, x):
        results = []

        x = self.conv_1(x)

        for layer in self.layer_1:
            x = layer(x)
        results.append(x)

        for layer in self.layer_2:
            x = layer(x)
        results.append(x)

        for layer in self.layer_3:
            x = layer(x)
        results.append(x)

        for layer in self.layer_4:
            x = layer(x)
        results.append(x)

        for layer in self.layer_5:
            x = layer(x)
        results.append(x)

        return results


_TIMM_TO_LIMR = {
    "conv_1.0.": "stem.conv.",
    "conv_1.1.": "stem.bn.",

    "layer_1.0.block.exp_1x1":      "stages.0.0.conv1_1x1.conv",
    "layer_1.0.block.exp_1x1_bn":   "stages.0.0.conv1_1x1.bn",
    "layer_1.0.block.conv_3x3":     "stages.0.0.conv2_kxk.conv",
    "layer_1.0.block.conv_3x3_bn":  "stages.0.0.conv2_kxk.bn",
    "layer_1.0.block.red_1x1":      "stages.0.0.conv3_1x1.conv",
    "layer_1.0.block.red_1x1_bn":   "stages.0.0.conv3_1x1.bn",

    "layer_2.0.block.exp_1x1":      "stages.1.0.conv1_1x1.conv",
    "layer_2.0.block.exp_1x1_bn":   "stages.1.0.conv1_1x1.bn",
    "layer_2.0.block.conv_3x3":     "stages.1.0.conv2_kxk.conv",
    "layer_2.0.block.conv_3x3_bn":  "stages.1.0.conv2_kxk.bn",
    "layer_2.0.block.red_1x1":      "stages.1.0.conv3_1x1.conv",
    "layer_2.0.block.red_1x1_bn":   "stages.1.0.conv3_1x1.bn",
    "layer_2.1.block.exp_1x1":      "stages.1.1.conv1_1x1.conv",
    "layer_2.1.block.exp_1x1_bn":   "stages.1.1.conv1_1x1.bn",
    "layer_2.1.block.conv_3x3":     "stages.1.1.conv2_kxk.conv",
    "layer_2.1.block.conv_3x3_bn":  "stages.1.1.conv2_kxk.bn",
    "layer_2.1.block.red_1x1":      "stages.1.1.conv3_1x1.conv",
    "layer_2.1.block.red_1x1_bn":   "stages.1.1.conv3_1x1.bn",

    "layer_3.0.block.exp_1x1":      "stages.2.0.conv1_1x1.conv",
    "layer_3.0.block.exp_1x1_bn":   "stages.2.0.conv1_1x1.bn",
    "layer_3.0.block.conv_3x3":     "stages.2.0.conv2_kxk.conv",
    "layer_3.0.block.conv_3x3_bn":  "stages.2.0.conv2_kxk.bn",
    "layer_3.0.block.red_1x1":      "stages.2.0.conv3_1x1.conv",
    "layer_3.0.block.red_1x1_bn":   "stages.2.0.conv3_1x1.bn",
    "layer_3.1.local_rep.0.0":      "stages.2.1.conv_kxk.conv",
    "layer_3.1.local_rep.0.1":      "stages.2.1.conv_kxk.bn",
    "layer_3.1.local_rep.1":        "stages.2.1.conv_1x1",
    "layer_3.1.global_rep.0.pre_norm_attn.0": "stages.2.1.transformer.0.norm1",
    "layer_3.1.global_rep.0.pre_norm_attn.1": "stages.2.1.transformer.0.attn",
    "layer_3.1.global_rep.0.pre_norm_ffn.0":  "stages.2.1.transformer.0.norm2",
    "layer_3.1.global_rep.0.pre_norm_ffn.1":  "stages.2.1.transformer.0.mlp.fc1",
    "layer_3.1.global_rep.0.pre_norm_ffn.4":  "stages.2.1.transformer.0.mlp.fc2",
    "layer_3.1.global_rep.1.pre_norm_attn.0": "stages.2.1.transformer.1.norm1",
    "layer_3.1.global_rep.1.pre_norm_attn.1": "stages.2.1.transformer.1.attn",
    "layer_3.1.global_rep.1.pre_norm_ffn.0":  "stages.2.1.transformer.1.norm2",
    "layer_3.1.global_rep.1.pre_norm_ffn.1":  "stages.2.1.transformer.1.mlp.fc1",
    "layer_3.1.global_rep.1.pre_norm_ffn.4":  "stages.2.1.transformer.1.mlp.fc2",
    "layer_3.1.global_rep.2":         "stages.2.1.norm",
    "layer_3.1.conv_proj.0":          "stages.2.1.conv_proj.conv",
    "layer_3.1.conv_proj.1":          "stages.2.1.conv_proj.bn",

    "layer_4.0.block.exp_1x1":      "stages.3.0.conv1_1x1.conv",
    "layer_4.0.block.exp_1x1_bn":   "stages.3.0.conv1_1x1.bn",
    "layer_4.0.block.conv_3x3":     "stages.3.0.conv2_kxk.conv",
    "layer_4.0.block.conv_3x3_bn":  "stages.3.0.conv2_kxk.bn",
    "layer_4.0.block.red_1x1":      "stages.3.0.conv3_1x1.conv",
    "layer_4.0.block.red_1x1_bn":   "stages.3.0.conv3_1x1.bn",
    "layer_4.1.local_rep.0.0":      "stages.3.1.conv_kxk.conv",
    "layer_4.1.local_rep.0.1":      "stages.3.1.conv_kxk.bn",
    "layer_4.1.local_rep.1":        "stages.3.1.conv_1x1",
    "layer_4.1.global_rep.0.pre_norm_attn.0": "stages.3.1.transformer.0.norm1",
    "layer_4.1.global_rep.0.pre_norm_attn.1": "stages.3.1.transformer.0.attn",
    "layer_4.1.global_rep.0.pre_norm_ffn.0":  "stages.3.1.transformer.0.norm2",
    "layer_4.1.global_rep.0.pre_norm_ffn.1":  "stages.3.1.transformer.0.mlp.fc1",
    "layer_4.1.global_rep.0.pre_norm_ffn.4":  "stages.3.1.transformer.0.mlp.fc2",
    "layer_4.1.global_rep.1.pre_norm_attn.0": "stages.3.1.transformer.1.norm1",
    "layer_4.1.global_rep.1.pre_norm_attn.1": "stages.3.1.transformer.1.attn",
    "layer_4.1.global_rep.1.pre_norm_ffn.0":  "stages.3.1.transformer.1.norm2",
    "layer_4.1.global_rep.1.pre_norm_ffn.1":  "stages.3.1.transformer.1.mlp.fc1",
    "layer_4.1.global_rep.1.pre_norm_ffn.4":  "stages.3.1.transformer.1.mlp.fc2",
    "layer_4.1.global_rep.2.pre_norm_attn.0": "stages.3.1.transformer.2.norm1",
    "layer_4.1.global_rep.2.pre_norm_attn.1": "stages.3.1.transformer.2.attn",
    "layer_4.1.global_rep.2.pre_norm_ffn.0":  "stages.3.1.transformer.2.norm2",
    "layer_4.1.global_rep.2.pre_norm_ffn.1":  "stages.3.1.transformer.2.mlp.fc1",
    "layer_4.1.global_rep.2.pre_norm_ffn.4":  "stages.3.1.transformer.2.mlp.fc2",
    "layer_4.1.global_rep.3.pre_norm_attn.0": "stages.3.1.transformer.3.norm1",
    "layer_4.1.global_rep.3.pre_norm_attn.1": "stages.3.1.transformer.3.attn",
    "layer_4.1.global_rep.3.pre_norm_ffn.0":  "stages.3.1.transformer.3.norm2",
    "layer_4.1.global_rep.3.pre_norm_ffn.1":  "stages.3.1.transformer.3.mlp.fc1",
    "layer_4.1.global_rep.3.pre_norm_ffn.4":  "stages.3.1.transformer.3.mlp.fc2",
    "layer_4.1.global_rep.4":         "stages.3.1.norm",
    "layer_4.1.conv_proj.0":          "stages.3.1.conv_proj.conv",
    "layer_4.1.conv_proj.1":          "stages.3.1.conv_proj.bn",

    "layer_5.0.block.exp_1x1":      "stages.4.0.conv1_1x1.conv",
    "layer_5.0.block.exp_1x1_bn":   "stages.4.0.conv1_1x1.bn",
    "layer_5.0.block.conv_3x3":     "stages.4.0.conv2_kxk.conv",
    "layer_5.0.block.conv_3x3_bn":  "stages.4.0.conv2_kxk.bn",
    "layer_5.0.block.red_1x1":      "stages.4.0.conv3_1x1.conv",
    "layer_5.0.block.red_1x1_bn":   "stages.4.0.conv3_1x1.bn",
    "layer_5.1.local_rep.0.0":      "stages.4.1.conv_kxk.conv",
    "layer_5.1.local_rep.0.1":      "stages.4.1.conv_kxk.bn",
    "layer_5.1.local_rep.1":        "stages.4.1.conv_1x1",
    "layer_5.1.global_rep.0.pre_norm_attn.0": "stages.4.1.transformer.0.norm1",
    "layer_5.1.global_rep.0.pre_norm_attn.1": "stages.4.1.transformer.0.attn",
    "layer_5.1.global_rep.0.pre_norm_ffn.0":  "stages.4.1.transformer.0.norm2",
    "layer_5.1.global_rep.0.pre_norm_ffn.1":  "stages.4.1.transformer.0.mlp.fc1",
    "layer_5.1.global_rep.0.pre_norm_ffn.4":  "stages.4.1.transformer.0.mlp.fc2",
    "layer_5.1.global_rep.1.pre_norm_attn.0": "stages.4.1.transformer.1.norm1",
    "layer_5.1.global_rep.1.pre_norm_attn.1": "stages.4.1.transformer.1.attn",
    "layer_5.1.global_rep.1.pre_norm_ffn.0":  "stages.4.1.transformer.1.norm2",
    "layer_5.1.global_rep.1.pre_norm_ffn.1":  "stages.4.1.transformer.1.mlp.fc1",
    "layer_5.1.global_rep.1.pre_norm_ffn.4":  "stages.4.1.transformer.1.mlp.fc2",
    "layer_5.1.global_rep.2.pre_norm_attn.0": "stages.4.1.transformer.2.norm1",
    "layer_5.1.global_rep.2.pre_norm_attn.1": "stages.4.1.transformer.2.attn",
    "layer_5.1.global_rep.2.pre_norm_ffn.0":  "stages.4.1.transformer.2.norm2",
    "layer_5.1.global_rep.2.pre_norm_ffn.1":  "stages.4.1.transformer.2.mlp.fc1",
    "layer_5.1.global_rep.2.pre_norm_ffn.4":  "stages.4.1.transformer.2.mlp.fc2",
    "layer_5.1.global_rep.3":         "stages.4.1.norm",
    "layer_5.1.conv_proj.0":          "stages.4.1.conv_proj.conv",
    "layer_5.1.conv_proj.1":          "stages.4.1.conv_proj.bn",
    "head.": "head.",
}

_LIMR_TO_ANOMALIB = {
    "conv_1": "stem",
    "layer_1": "stage0",
    "layer_2": "stage1",
    "layer_3": "stage2",
    "layer_4": "stage3",
    "layer_5": "stage4",
}


class LiMREncoder(nn.Module):
    """LiMR encoder matching original architecture.

    Splits MobileViTv2 into independently-managed stages.
    Masking is applied inside the LiMViT block (stage2/layer_3),
    consistent with the original paper's design.

    Args:
        width_multiplier: model width multiplier (alpha).
        mask_ratio: training mask ratio (0 to disable masking).
        frozen_stages: number of stages to freeze (1=stem,
            2=stem+stage0, 3=stem+stage0+stage1).
        block_ffn_dropout: dropout in FFN layers.
        block_attn_dropout: dropout in attention layers.
        block_dropout: dropout in MobileViTBlockv2.
            Defaults to ``0.1``.
        load_timm_weights: if True, automatically load timm pretrained weights.
    """

    def __init__(
        self,
        width_multiplier: float = 1.75,
        mask_ratio: float = 0.4,
        frozen_stages: int = 3,
        block_ffn_dropout: float = 0.1,
        block_attn_dropout: float = 0.0,
        block_dropout: float = 0.1,
        load_timm_weights: bool = True,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio

        mobilevit = MobileViTv2(
            width_multiplier=width_multiplier,
            block_ffn_dropout=block_ffn_dropout,
            block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
        )

        self.stem = mobilevit.conv_1
        self.stage0 = mobilevit.layer_1
        self.stage1 = mobilevit.layer_2
        self.stage2 = mobilevit.layer_3
        self.stage3 = mobilevit.layer_4
        self.stage4 = mobilevit.layer_5

        self._freeze_stages(frozen_stages)

        if load_timm_weights:
            self._load_timm_weights(width_multiplier)

    def _generate_mask(self, x):
        """Generate mask at 7x7 semantic grid (matching original LiMR).
        
        Uses random_masking from masking.py at a fixed 7x7 grid, then
        nearest-neighbor upsamples to both the spatial feature resolution
        and the MobileViT patch resolution for idx_keep/ids_restore.
        """
        B, C, H, W = x.shape
        GRID = 7  # fixed semantic grid matching original random_masking

        # 1. Generate base mask at 7x7 grid via masking.py
        _, mask_base, _ = random_masking(x, self.mask_ratio, grid_size=GRID)
        # mask_base: (B, 49), 0=keep, 1=remove

        # 2. Upsample to spatial mask (H x W)
        # Each 7x7 cell maps to H/GRID x W/GRID pixels
        mask_spatial_grid = mask_base.reshape(B, GRID, GRID).unsqueeze(1)  # (B, 1, 7, 7)
        mask_spatial = torch.repeat_interleave(
            torch.repeat_interleave(mask_spatial_grid, H // GRID, dim=2),
            W // GRID, dim=3,
        )  # (B, 1, H, W), 0=keep, 1=remove

        # 3. Upsample to MobileViT patch grid and generate patch-level idx_keep/ids_restore
        # MobileViTBlockv2 uses patch_h=patch_w=2 (matching _generate_mask's call site)
        PH, PW = 2, 2
        HP, WP = H // PH, W // PW  # patch grid dimensions, e.g. 14 x 14
        upscale_patch = HP // GRID  # e.g. 14 // 7 = 2

        mask_patchgrid = mask_base.reshape(B, GRID, GRID)
        mask_patchgrid = torch.repeat_interleave(
            torch.repeat_interleave(mask_patchgrid, upscale_patch, dim=1),
            upscale_patch, dim=2,
        )  # (B, HP, WP), 0=keep, 1=remove

        # Flatten to (B, HP*WP) and derive idx_keep/ids_restore
        mask_flat = mask_patchgrid.reshape(B, HP * WP)
        N_patches = HP * WP
        len_keep = int(N_patches * (1 - self.mask_ratio))

        # argsort: 0 (keep) come first, 1 (remove) come last
        ids_shuffle = torch.argsort(mask_flat.float(), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        idx_keep = ids_shuffle[:, :len_keep]

        # Flip spatial mask polarity: forward_masked expects 1=keep, 0=remove
        mask_spatial = 1 - mask_spatial

        return mask_spatial, idx_keep, ids_restore

    def forward(self, x):
        feats = []

        x = self.stem(x)

        for layer in self.stage0:
            x = layer(x)
        feats.append(x)

        for layer in self.stage1:
            x = layer(x)
        feats.append(x)

        if self.training and self.mask_ratio > 0:
            x = self.stage2[0](x)
            mask_spatial, idx_keep, ids_restore = self._generate_mask(x)
            x = self.stage2[1].forward_masked(x, mask_spatial, idx_keep, ids_restore)
        else:
            for layer in self.stage2:
                x = layer(x)
        feats.append(x)

        for layer in self.stage3:
            x = layer(x)
        feats.append(x)

        for layer in self.stage4:
            x = layer(x)
        feats.append(x)

        return feats

    def _freeze_stages(self, n):
        stage_names = {1: ["stem"], 2: ["stem", "stage0"], 3: ["stem", "stage0", "stage1"]}
        modules = {
            1: [self.stem],
            2: [self.stem, self.stage0],
            3: [self.stem, self.stage0, self.stage1],
        }
        frozen = 0
        for m_list in modules.get(n, []):
            for p in m_list.parameters():
                p.requires_grad = False
                frozen += 1
        logger.info("Frozen stages: %d -> %s (%d params frozen)", n, stage_names.get(n, []), frozen)

    def _load_timm_weights(self, alpha):
        try:
            import timm
        except ImportError:
            logger.warning("timm not installed, skipping pretrained weight loading")
            return

        model_name = f"mobilevitv2_{int(alpha * 100):03d}"
        timm_model = timm.create_model(model_name, pretrained=True)
        timm_state = timm_model.state_dict()
        del timm_model  # release timm model immediately to avoid doubling encoder memory

        logger.info("Loading pretrained weights from timm model '%s' (alpha=%.2f)", model_name, alpha)

        remapped = {}
        for k, v in timm_state.items():
            new_k = k
            for limr_key, timm_prefix in _TIMM_TO_LIMR.items():
                if k.startswith(timm_prefix):
                    new_k = k.replace(timm_prefix, limr_key)
                    break
            if new_k != k:
                remapped[new_k] = v

        if len(remapped) == 0:
            raise RuntimeError(
                "Weight remapping produced 0 keys. "
                "The _TIMM_TO_LIMR mapping table may be outdated "
                "for timm model '%s'." % model_name
            )

        final = {}
        for k, v in remapped.items():
            new_k = k
            for old_prefix, new_prefix in _LIMR_TO_ANOMALIB.items():
                if k.startswith(old_prefix):
                    new_k = k.replace(old_prefix, new_prefix, 1)
                    break
            final[new_k] = v

        missing, unexpected = self.load_state_dict(final, strict=False)
        loaded = len(final) - len(unexpected)

        logger.info(
            "Weight transfer: %d timm keys remapped, %d loaded, %d missing, %d unexpected",
            len(final), loaded, len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Keys not loaded (missing in src): %s", missing)
        if unexpected:
            logger.warning("Keys not found in dst: %s", unexpected)
        if loaded < len(final) * 0.5:
            raise RuntimeError(
                "Only %d/%d keys loaded. The weight mapping may be broken." % (loaded, len(final))
            )
