import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import numpy as np

from anomalib.models.components import AnomalibModule

from .torch_model import LiMRModel
from .components.losses import LiMRLoss, cal_anomaly_map


class WarmupCosineScheduler(CosineAnnealingLR):
    def __init__(self, warmup_epochs, **kwargs):
        self.warmup_epochs = warmup_epochs
        super().__init__(**kwargs)
        self.base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        return super().get_lr()


class LiMR(AnomalibModule):
    def __init__(
        self,
        backbone="resnet50",
        layers_to_extract_from=None,
        alpha=1.0,
        mask_ratio=0.75,
        test_mask_ratio=0.0,
        scale_factors=None,
        fpn_output_dim=None,
        lr=0.001,
        weight_decay=0.05,
        warmup_epochs=15,
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

        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.test_mask_ratio = test_mask_ratio
        self.scale_factors = scale_factors
        self.fpn_output_dim = fpn_output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.model = LiMRModel(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            alpha=alpha,
            scale_factors=scale_factors,
            fpn_output_dim=fpn_output_dim,
            block_ffn_dropout=block_ffn_dropout,
            block_attn_dropout=block_attn_dropout,
        )

        self.loss = LiMRLoss()

    def _get_trainable_params(self, weight_decay):
        wd_params = []
        no_wd_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.endswith(".bias"):
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        return [
            {"params": wd_params, "weight_decay": weight_decay},
            {"params": no_wd_params, "weight_decay": 0},
        ]

    def training_step(self, batch, *args, **kwargs):
        x = batch["image"]

        teacher_feats, student_feats = self.model.forward_train(x, mask_ratio=self.mask_ratio)

        loss = self.loss(teacher_feats, student_feats)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        x = batch["image"]

        teacher_feats, student_feats = self.model.forward_eval(x, mask_ratio=self.test_mask_ratio)

        anomaly_maps, _ = cal_anomaly_map(
            teacher_feats, student_feats,
            out_size=x.shape[-1],
            amap_mode="a",
        )

        for item in range(len(anomaly_maps)):
            anomaly_maps[item] = gaussian_filter(anomaly_maps[item], sigma=4)

        pred_scores = np.max(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1)

        batch["anomaly_maps"] = torch.from_numpy(anomaly_maps).float()
        batch["pred_scores"] = torch.from_numpy(pred_scores).float()

        return batch

    def configure_optimizers(self):
        trainable_params = self._get_trainable_params(self.weight_decay)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            T_max=self.trainer.max_epochs - self.warmup_epochs if self.trainer else 200 - self.warmup_epochs,
            eta_min=1e-5,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
