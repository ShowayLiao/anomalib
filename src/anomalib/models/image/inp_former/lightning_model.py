# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for the INP-Former model.

This module implements the Lightning interface for the INP-Former model,
providing training, validation, and optimization functionality.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn.init import trunc_normal_
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
from anomalib.models.image.inp_former.torch_model import INP_FormerModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

logger = logging.getLogger(__name__)

# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 1e-3,
        "final_value": 1e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}


class INP_Former(AnomalibModule):
    """INP-Former Lightning Module for anomaly detection.

    This lightning module trains the INP-Former model, which extracts intrinsic
    normal prototypes from test images to detect anomalies.

    Args:
        encoder_name (str): Name of the DINOv2 encoder to use.
        inp_num (int): Number of intrinsic normal prototypes.
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP.
        decoder_depth (int): Number of decoder layers.
        target_layers (list[int] | None): List of encoder layers to extract features from.
        fuse_layer_encoder (list[list[int]] | None): Layer groupings for encoder feature fusion.
        fuse_layer_decoder (list[list[int]] | None): Layer groupings for decoder feature fusion.
        remove_class_token (bool): Whether to remove class token from features.
        encoder_require_grad_layer (list[int]): List of encoder layers to keep trainable.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or flag.
        post_processor (PostProcessor | bool, optional): Post-processor instance or flag.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        inp_num: int = 6,
        bottleneck_dropout: float = 0.0,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        encoder_require_grad_layer: list[int] = [],
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

        # Initialize the INP-Former model
        self.model: INP_FormerModel = INP_FormerModel(
            encoder_name=encoder_name,
            inp_num=inp_num,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
            encoder_require_grad_layer=encoder_require_grad_layer,
        )

        # Set trainable parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze trainable modules
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.aggregation.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.prototype_token:
            param.requires_grad = True

        # Collect trainable modules
        self.trainable_modules = torch.nn.ModuleList([
            self.model.bottleneck,
            self.model.aggregation,
            self.model.decoder,
        ])
        # Add prototype tokens to trainable parameters
        self.trainable_params = list(self.trainable_modules.parameters())
        for token in self.model.prototype_token:
            self.trainable_params.append(token)

        # Initialize trainable modules
        self._initialize_trainable_modules()

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        crop_size: int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for INP-Former.

        Args:
            image_size (tuple[int, int] | None): Target size for image resizing.
            crop_size (int | None): Target size for center cropping.

        Returns:
            PreProcessor: Configured pre-processor.
        """
        crop_size = crop_size or DEFAULT_CROP_SIZE
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        if crop_size > min(image_size):
            msg = f"Crop size {crop_size} cannot be larger than image size {image_size}"
            raise ValueError(msg)

        data_transforms = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for the INP-Former model.

        Args:
            batch (Batch): Input batch containing images.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the computed loss.
        """
        del args, kwargs  # Unused
        loss = self.model(batch.image, global_step=self.global_step)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step for the INP-Former model.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Updated batch with predictions.
        """
        del args, kwargs  # Unused
        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler.

        Returns:
            OptimizerLRScheduler: Tuple containing optimizer and scheduler configurations.
        """
        # Determine total training steps
        max_epochs = getattr(self.trainer, "max_epochs", -1)
        max_steps = getattr(self.trainer, "max_steps", -1)

        if max_epochs is None:
            max_epochs = -1
        if max_steps is None:
            max_steps = -1

        if max_epochs < 0 and max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        if max_epochs < 0:
            total_steps = max_steps
        elif max_steps < 0:
            total_steps = max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            total_steps = min(max_steps, max_epochs * len(self.trainer.datamodule.train_dataloader()))

        # Configure optimizer
        optimizer_config = TRAINING_CONFIG["optimizer"]
        optimizer = StableAdamW([{"params": self.trainable_params}], **optimizer_config)

        # Configure scheduler
        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        scheduler_config["total_iters"] = total_steps
        lr_scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Always returns LearningType.ONE_CLASS for unsupervised learning.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return INP-Former-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments.
        """
        trainer_config = TRAINING_CONFIG["trainer"].copy()
        # Remove max_steps to allow user override
        trainer_config.pop("max_steps", None)
        return trainer_config

    def _initialize_trainable_modules(self) -> None:
        """Initialize trainable modules with truncated normal initialization."""
        # Initialize MLP layers in bottleneck, aggregation, and decoder
        for module in self.trainable_modules.modules():
            if isinstance(module, torch.nn.Linear):
                trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)