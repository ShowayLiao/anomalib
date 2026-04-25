from torch import nn
from typing import Dict, Tuple, Optional

from .base_layers import InvertedResidual
from .mobilevit_v2_block import MobileViTBlockv2 as Block
from .model_utils import bound_fn, make_divisible


class MobileViTv2(nn.Module):
    def __init__(self, width_multiplier, block_ffn_dropout=0.1, block_attn_dropout=0.0) -> None:
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
        )

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
            block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
        )

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
            block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
        )

    def _make_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False,
        block_ffn_dropout=0.1, block_attn_dropout=0.0,
    ) -> Tuple[nn.ModuleList, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate,
                block_ffn_dropout=block_ffn_dropout, block_attn_dropout=block_attn_dropout,
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
            mask_block = False

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                mask_block=mask_block,
            )

            block.append(layer)
            input_channel = output_channels

        return block, input_channel

    def _make_mit_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False,
        block_ffn_dropout=0.1, block_attn_dropout=0.0,
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
                mask_block=False,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = 0.0

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

    def forward(self, x, masks=None, ids_keep_list=None, ids_restore_list=None):
        results = []

        x = self.conv_1(x)

        for layer in self.layer_1:
            x = layer(x, masks[0] if masks is not None else None)
        results.append(x)

        for layer in self.layer_2:
            x = layer(x, masks[1] if masks is not None else None)
        results.append(x)

        for layer in self.layer_3:
            if masks is not None and ids_keep_list is not None and ids_restore_list is not None:
                x = layer(x, masks[2], ids_keep_list[2], ids_restore_list[2])
            else:
                x = layer(x, None, None, None)
        results.append(x)

        for layer in self.layer_4:
            x = layer(x, None, None, None)
        results.append(x)

        for layer in self.layer_5:
            x = layer(x, None, None, None)
        results.append(x)

        return results
