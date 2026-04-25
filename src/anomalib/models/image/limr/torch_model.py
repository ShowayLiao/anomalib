from functools import partial
import torch
import torch.nn as nn

from anomalib.models.components.feature_extractors import TimmFeatureExtractor

from .components.encoder import MobileViTv2
from .components.decoder import FPN
from .components.masking import mask_everylayer, get_2d_sincos_pos_embed


class LiMRModel(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        layers_to_extract_from=None,
        alpha=1.0,
        scale_factors=None,
        fpn_output_dim=None,
        block_ffn_dropout=0.1,
        block_attn_dropout=0.0,
    ):
        super().__init__()

        if layers_to_extract_from is None:
            layers_to_extract_from = ["layer1", "layer2", "layer3"]
        if scale_factors is None:
            scale_factors = (4.0, 2.0, 1.0)
        if fpn_output_dim is None:
            fpn_output_dim = (64, 128, 256, 512)

        self.layers_to_extract_from = layers_to_extract_from

        # Teacher model (frozen)
        self.teacher = TimmFeatureExtractor(
            backbone=backbone,
            pre_trained=True,
            layers=layers_to_extract_from,
            requires_grad=False,
        )

        # Student encoder
        self.encoder = MobileViTv2(
            width_multiplier=alpha,
            block_ffn_dropout=block_ffn_dropout,
            block_attn_dropout=block_attn_dropout,
        )

        # Decoder FPN
        embed_dim = [64, 128, 256, 384, 512]
        embed_dim = [int(x * alpha) for x in embed_dim]
        decoder_embed_dim = embed_dim[3]

        self.decoder_FPN_pos_embed = nn.Parameter(
            torch.zeros(1, decoder_embed_dim, 14, 14),
            requires_grad=False,
        )

        self.decoder = FPN(embed_dim[::-1], fpn_output_dim[::-1], 4)

        self._initialize_weights(decoder_embed_dim)

    def _initialize_weights(self, decoder_embed_dim):
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, 14, cls_token=False)
        decoder_pos_embed = decoder_pos_embed.reshape(14, 14, -1)
        decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float()
        decoder_pos_embed = decoder_pos_embed.permute(2, 0, 1)
        decoder_pos_embed = decoder_pos_embed.unsqueeze(0)
        self.decoder_FPN_pos_embed.data.copy_(decoder_pos_embed)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio):
        masks, ids_keep_list, ids_restore_list = mask_everylayer(x, mask_ratio)
        layers = self.encoder(x, masks, ids_keep_list, ids_restore_list)
        return layers, masks, ids_restore_list

    def forward_decoder(self, x, mask=None):
        results = self.decoder(x, mask)
        results = results[1:]
        return {
            layer: feature
            for layer, feature in zip(self.layers_to_extract_from, results[::-1])
        }

    def forward_train(self, x, mask_ratio=0.75):
        with torch.no_grad():
            teacher_features = self.teacher(x)

        latent, mask, _ = self.forward_encoder(x, mask_ratio)
        student_features = self.forward_decoder(latent, mask)

        teacher_feat_list = [teacher_features[key] for key in self.layers_to_extract_from]
        student_feat_list = [student_features[key] for key in self.layers_to_extract_from]

        return teacher_feat_list, student_feat_list

    def forward_eval(self, x, mask_ratio=0.0):
        with torch.no_grad():
            teacher_features = self.teacher(x)

        latent, mask, _ = self.forward_encoder(x, mask_ratio)
        student_features = self.forward_decoder(latent, mask)

        teacher_feat_list = [teacher_features[key] for key in self.layers_to_extract_from]
        student_feat_list = [student_features[key] for key in self.layers_to_extract_from]

        return teacher_feat_list, student_feat_list
