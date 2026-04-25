# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""INP-Former: Intrinsic Normal Prototypes for Universal Anomaly Detection.

This module implements the INP-Former model for anomaly detection, which extracts
intrinsic normal prototypes from test images to detect anomalies. The model leverages
DINOv2 pre-trained features and a novel attention mechanism to capture normal patterns.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import INP_Former
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = INP_Former()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)
    >>> predictions = engine.predict(model, datamodule=datamodule)

Notes:
    - Uses DINOv2 Vision Transformer as the backbone encoder
    - Extracts intrinsic normal prototypes from test images
    - Employs INP-Guided Attention for efficient reconstruction
    - Supports both anomaly detection and localization tasks
"""

from anomalib.models.image.inp_former.lightning_model import INP_Former

__all__ = ["INP_Former"]