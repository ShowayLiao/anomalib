"""AeBAD-V Data Module.

This module provides a PyTorch Lightning DataModule for the AeBAD-V dataset
(video frames with domain shift).

Dataset Structure:
    AeBAD_V/
    ├── train/good/{video1_train, video2_train, video3_train, video4_train}/
    └── test/
        ├── video1/{good,anomaly}/
        ├── video2/{good,anomaly}/
        └── video3/{good,anomaly}/

Note:
    This dataset does not provide pixel-level ground truth masks. Only
    image-level classification metrics (AUROC) are applicable.
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.aebad_v import AeBADVDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from anomalib.utils.path import resolve_with_warning

logger = logging.getLogger(__name__)


class AeBAD_V(AnomalibDataModule):
    """AeBAD-V Datamodule for video-based wind blade anomaly detection.

    Args:
        root: Path to root directory containing the AeBAD_V dataset.
            Defaults to ``"./datasets/AeBAD/AeBAD_V"``.
        category: Category name. Defaults to ``"AeBAD_V"``.
        domain_shift: Domain shift category for test split
            (``"video1"``, ``"video2"``, ``"video3"``).
            Defaults to ``"video1"``.
        image_size: Target image size ``(H, W)``. Defaults to ``(256, 256)``.
        train_batch_size: Training batch size. Defaults to ``32``.
        eval_batch_size: Evaluation batch size. Defaults to ``32``.
        num_workers: Number of dataloader workers. Defaults to ``8``.
        train_augmentations: Augmentations for training images.
        val_augmentations: Augmentations for validation images.
        test_augmentations: Augmentations for test images.
        augmentations: General augmentations if stage-specific not provided.
        test_split_mode: Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        val_split_mode: Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        seed: Seed for reproducibility.
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/AeBAD/AeBAD_V",
        category: str = "AeBAD_V",
        domain_shift: str = "video1",
        image_size: tuple[int, int] = (256, 256),
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            val_split_mode=val_split_mode,
            seed=seed,
        )

        root = resolve_with_warning(root, "AeBAD_V")
        self.root = Path(root)
        self.category = category
        self.domain_shift = domain_shift
        self.image_size = image_size

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = AeBADVDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = AeBADVDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
            domain_shift=self.domain_shift,
        )

    def prepare_data(self) -> None:
        if self.root.is_dir():
            logger.info("Found the AeBAD_V dataset.")
        else:
            msg = (
                f"AeBAD_V dataset not found at {self.root}. "
                "Please download the dataset manually."
            )
            raise FileNotFoundError(msg)
