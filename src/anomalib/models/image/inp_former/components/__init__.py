# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for INP-Former model.

This module provides all the necessary components for the INP-Former Vision Transformer
architecture including attention mechanisms, transformer blocks, MLP layers, and loss functions.
"""

# Layer components
from .attention import Aggregation_Attention, Prototype_Attention
from .blocks import Aggregation_Block, Prototype_Block
from .mlp import Mlp
from .losses import INPCoherenceLoss, SoftMiningLoss

__all__ = [
    # Attention mechanisms
    "Aggregation_Attention",
    "Prototype_Attention",
    # Transformer blocks
    "Aggregation_Block",
    "Prototype_Block",
    # MLP
    "Mlp",
    # Loss functions
    "INPCoherenceLoss",
    "SoftMiningLoss",
]