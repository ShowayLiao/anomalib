# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MLP layer implementation for INP-Former model.

This module provides the MLP layer used in Transformer blocks for both the
Aggregation and Prototype blocks in the INP-Former architecture.
"""

from torch import nn


class Mlp(nn.Module):
    """MLP layer for Transformer blocks.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to None.
        out_features (int, optional): Number of output features. Defaults to None.
        act_layer (callable, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: callable = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x