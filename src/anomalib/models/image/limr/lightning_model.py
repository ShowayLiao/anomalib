# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LiMR: Lightweight Masked Reconstruction for Anomaly Detection.

This module implements the LiMR model for anomaly detection. The model uses a teacher-student
architecture where a frozen teacher extracts features, and a lightweight student encoder-decoder
reconstructs masked semantic features. The reconstruction error is used to detect anomalies.

Semantic masked reconstruction is applied inside the LiMViT block (stage2/layer_3),
matching the original paper's design. Frozen initial stages extract semantic features,
while the LiMViT block performs masked reconstruction on these features internally.

Example:
    >>> from anomalib.models import LiMR
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = LiMR(backbone="resnet50", alpha=1.75, mask_ratio=0.4)

    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
"""

from collections.abc import Sequence
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from .components.losses import LiMRLoss
from .torch_model import LiMRModel


class WarmupCosineScheduler(optim.lr_scheduler.CosineAnnealingLR):
    """Cosine annealing LR scheduler with linear warmup."""

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
    """PL Lightning Module for LiMR Algorithm.

    Args:
        backbone: Backbone of CNN network for teacher.
            Defaults to ``"resnet50"``.
        layers_to_extract_from: Layers to extract features from the teacher backbone.
            Defaults to ``["layer1", "layer2", "layer3"]``.
        alpha: Width multiplier for the student encoder.
            Defaults to ``1.0``.
        mask_ratio: Ratio of patches to mask during training.
            Defaults to ``0.4``.
        test_mask_ratio: Ratio of patches to mask during evaluation.
            Defaults to ``0.0``.
        scale_factors: Scale factors for feature pyramid.
            Defaults to ``(4.0, 2.0, 1.0)``.
        fpn_output_dim: Output dimensions for FPN layers.
            Defaults to ``None`` (auto-detect from teacher).
        frozen_stages: Number of encoder stages to freeze (1=stem, 2=stem+stage0,
            3=stem+stage0+stage1).
            Defaults to ``3``.
        load_timm_weights: Whether to load timm pretrained MobileViTv2 weights.
            Defaults to ``True``.
        lr: Learning rate.
            Defaults to ``0.001``.
        weight_decay: Weight decay for optimizer.
            Defaults to ``0.05``.
        warmup_epochs: Number of warmup epochs for LR scheduler.
            Defaults to ``15``.
        block_ffn_dropout: Dropout rate for FFN blocks.
            Defaults to ``0.1``.
        block_attn_dropout: Dropout rate for attention blocks.
            Defaults to ``0.0``.
        block_dropout: Dropout rate for MobileViTBlockv2.
            Defaults to ``0.1``.
        pre_processor: Pre-processor for the model.
            Defaults to ``True``.
        post_processor: Post-processor instance.
            Defaults to ``True``.
        evaluator: Evaluator instance.
            Defaults to ``True``.
        visualizer: Visualizer instance.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layers_to_extract_from: Sequence[str] | None = None,
        alpha: float = 1.0,
        mask_ratio: float = 0.4,
        test_mask_ratio: float = 0.0,
        scale_factors: tuple[float, ...] | None = None,
        fpn_output_dim: tuple[int, ...] | None = None,
        frozen_stages: int = 3,
        load_timm_weights: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        warmup_epochs: int = 15,
        block_ffn_dropout: float = 0.1,
        block_attn_dropout: float = 0.0,
        block_dropout: float = 0.1,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        if layers_to_extract_from is None:
            layers_to_extract_from = ["layer1", "layer2", "layer3"]
        if scale_factors is None:
            scale_factors = (4.0, 2.0, 1.0)

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
            mask_ratio=mask_ratio,
            scale_factors=scale_factors,
            fpn_output_dim=fpn_output_dim,
            block_ffn_dropout=block_ffn_dropout,
            block_attn_dropout=block_attn_dropout,
            block_dropout=block_dropout,
            frozen_stages=frozen_stages,
            load_timm_weights=load_timm_weights,
        )

        # 供 Lightning ModelSummary 计算 FLOPs
        self.example_input_array = torch.randn(1, 3, 224, 224)

        self.loss = LiMRLoss()

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """LiMR preprocessor: Resize → CenterCrop → Normalize.

        Matches the original paper's pipeline: Resize to a larger size
        then CenterCrop to the target size.
        """
        image_size = image_size or (256, 256)
        crop_size = (224, 224)

        transform = Compose([
            Resize(image_size, antialias=True),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return LiMR trainer arguments."""
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

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

    def configure_optimizers(self) -> dict:
        """Configure optimizers and LR schedulers.

        Returns:
            dict: Optimizer and LR scheduler configuration.
        """
        trainable_params = self._get_trainable_params(self.weight_decay)

        optimizer = optim.AdamW(
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

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step.

        Args:
            batch: Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss.
        """
        del args, kwargs

        teacher_feats, student_feats = self.model(batch.image, mask_ratio=self.mask_ratio)
        loss = self.loss(teacher_feats, student_feats)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step.

        Args:
            batch: Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the batch with predictions.
        """
        del args, kwargs

        predictions = self.model(batch.image, mask_ratio=self.test_mask_ratio)
        return batch.update(**predictions._asdict())
