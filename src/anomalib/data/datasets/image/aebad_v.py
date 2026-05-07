"""AeBAD-V Dataset.

This module provides a PyTorch Dataset implementation for the AeBAD-V dataset
(video frames with domain shift). The dataset contains wind blade images captured
from multiple video sequences with normal and anomalous frames.

Dataset Structure:
    AeBAD_V/
    ├── train/good/
    │   ├── video1_train/
    │   ├── video2_train/
    │   ├── video3_train/
    │   └── video4_train/
    └── test/
        ├── video1/{good,anomaly}/
        ├── video2/{good,anomaly}/
        └── video3/{good,anomaly}/

Note:
    This dataset does not provide pixel-level ground truth masks. Only
    image-level classification metrics (AUROC) are applicable.

Reference:
    Zhang, Z., et al. "Industrial Anomaly Detection with Domain Shift:
    A Real-world Dataset and Masked Multi-scale Reconstruction."
    arXiv:2304.02216, 2023.
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG")


class AeBADVDataset(AnomalibDataset):
    """AeBAD-V dataset class for video-based wind blade anomaly detection.

    Args:
        root: Path to root directory containing the AeBAD_V dataset.
        category: Category name.
        domain_shift: Domain shift category for test split
            (``"video1"``, ``"video2"``, ``"video3"``).
        augmentations: Augmentations applied to input images.
        split: Dataset split - ``Split.TRAIN`` or ``Split.TEST``.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "AeBAD_V",
        domain_shift: str = "video1",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root = Path(root) if root is not None else Path("./datasets/AeBAD/AeBAD_V")
        self.category = category
        self.domain_shift = domain_shift
        self.split = split
        self.samples = make_aebad_v_dataset(
            root=self.root,
            split=self.split,
            domain_shift=self.domain_shift,
            extensions=IMG_EXTENSIONS,
        )


def make_aebad_v_dataset(
    root: str | Path,
    split: str | Split | None = None,
    domain_shift: str = "video1",
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create AeBAD-V samples by parsing the directory structure.

    Args:
        root: Path to AeBAD_V dataset root directory.
        split: Dataset split (``Split.TRAIN`` or ``Split.TEST``).
        domain_shift: Domain shift category for test split filtering
            (``"video1"``, ``"video2"``, ``"video3"``).
        extensions: Valid file extensions to include.

    Returns:
        DataFrame with columns: ``image_path``, ``mask_path``,
        ``label_index``, ``split``.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = []

    if split == Split.TRAIN:
        train_good_dir = root / "train" / "good"
        if not train_good_dir.is_dir():
            msg = f"Train directory not found: {train_good_dir}"
            raise FileNotFoundError(msg)

        for video_dir in train_good_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for img_path in video_dir.glob("*"):
                if img_path.suffix in extensions:
                    samples_list.append({
                        "image_path": str(img_path),
                        "mask_path": "",
                        "label_index": LabelName.NORMAL,
                        "split": Split.TRAIN,
                    })

    else:
        test_video_dir = root / "test" / domain_shift
        if not test_video_dir.is_dir():
            msg = f"Test directory not found: {test_video_dir}"
            raise FileNotFoundError(msg)

        for label_dir in test_video_dir.iterdir():
            if not label_dir.is_dir():
                continue
            is_normal = label_dir.name == "good"
            for img_path in label_dir.glob("*"):
                if img_path.suffix in extensions:
                    samples_list.append({
                        "image_path": str(img_path),
                        "mask_path": "",
                        "label_index": LabelName.NORMAL if is_normal else LabelName.ABNORMAL,
                        "split": Split.TEST,
                    })

    if not samples_list:
        msg = f"Found 0 images in {root} with split={split}, domain_shift={domain_shift}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    samples.attrs["task"] = "classification"

    return samples
