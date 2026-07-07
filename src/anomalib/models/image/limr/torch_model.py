import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

from anomalib.data import InferenceBatch
from anomalib.models.components.feature_extractors import TimmFeatureExtractor, dryrun_find_featuremap_dims

from .components.decoder import FPN
from .components.encoder import LiMREncoder
from .components.masking import get_2d_sincos_pos_embed


class LiMRModel(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        layers_to_extract_from=None,
        alpha=1.0,
        mask_ratio=0.4,
        scale_factors=None,
        fpn_output_dim=None,
        block_ffn_dropout=0.1,
        block_attn_dropout=0.0,
        block_dropout=0.1,
        frozen_stages=3,
        load_timm_weights=True,
    ):
        super().__init__()

        if layers_to_extract_from is None:
            layers_to_extract_from = ["layer1", "layer2", "layer3"]
        if scale_factors is None:
            scale_factors = (4.0, 2.0, 1.0)

        self.layers_to_extract_from = layers_to_extract_from
        self.mask_ratio = mask_ratio

        self.teacher = TimmFeatureExtractor(
            backbone=backbone,
            pre_trained=True,
            layers=layers_to_extract_from,
            requires_grad=False,
        )

        if fpn_output_dim is None:
            teacher_dims = dryrun_find_featuremap_dims(
                self.teacher, input_size=(256, 256), layers=layers_to_extract_from,
            )
            teacher_channels = [teacher_dims[layer]["num_features"] for layer in layers_to_extract_from]
            fpn_output_dim = (
                teacher_channels[0],
                teacher_channels[1],
                teacher_channels[2],
                teacher_channels[2] * 2,
            )

        self.encoder = LiMREncoder(
            width_multiplier=alpha,
            mask_ratio=mask_ratio,
            frozen_stages=frozen_stages,
            block_ffn_dropout=block_ffn_dropout,
            block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
            load_timm_weights=load_timm_weights,
        )

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

    def forward_decoder(self, x, mask=None):
        results = self.decoder(x, mask)
        results = results[1:]
        return {
            layer: feature
            for layer, feature in zip(self.layers_to_extract_from, results[::-1])
        }

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None) -> tuple | InferenceBatch:
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
            self.encoder.mask_ratio = mask_ratio

        with torch.no_grad():
            teacher_features = self.teacher(x)

        latent = self.encoder(x)
        student_features = self.forward_decoder(latent, None)

        teacher_feat_list = [teacher_features[key] for key in self.layers_to_extract_from]
        student_feat_list = [student_features[key] for key in self.layers_to_extract_from]

        if self.training:
            return teacher_feat_list, student_feat_list

        anomaly_map = self._compute_anomaly_map(teacher_feat_list, student_feat_list, out_size=x.shape[-1])
        pred_score = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1).values
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    @staticmethod
    def _compute_anomaly_map(teacher_feats, student_feats, out_size: int) -> torch.Tensor:
        anomaly_maps_list = []
        for i in range(len(student_feats)):
            a_map = 1 - F.cosine_similarity(student_feats[i], teacher_feats[i])
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            a_map = a_map.squeeze(1)
            anomaly_maps_list.append(a_map)

        anomaly_map = torch.zeros_like(anomaly_maps_list[0])
        for a_map in anomaly_maps_list:
            anomaly_map += a_map

        anomaly_map = anomaly_map.unsqueeze(1)
        anomaly_map = kornia.filters.gaussian_blur2d(
            anomaly_map, kernel_size=(33, 33), sigma=(4.0, 4.0),
            border_type="replicate",
        )
        anomaly_map = anomaly_map.squeeze(1)
        return anomaly_map
