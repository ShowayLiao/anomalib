"""AeBAD-S Dataset.

This module provides a PyTorch Dataset implementation for the AeBAD-S dataset
(static images with domain shift). The dataset contains wind blade images with
multiple defect types and pixel-level ground truth masks.

Dataset Structure:
    AeBAD_S/
    ├── train/good/
    │   ├── background/     # Normal images under background variation
    │   ├── illumination/   # Normal images under illumination variation
    │   └── view/           # Normal images under viewpoint variation
    ├── test/
    │   ├── good/
    │   │   └── same/       # Normal test images (no domain shift)
    │   ├── ablation/{same,view}/
    │   ├── breakdown/{same}/
    │   ├── fracture/{same,view}/
    │   └── groove/{same}/
    └── ground_truth/
        └── {defect}/{domain_shift}/{filename}.png  # Pixel-level masks

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
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".png", ".PNG")


class AeBADSDataset(AnomalibDataset):
    """AeBAD-S dataset class for static wind blade anomaly detection.

    Args:
        root: Path to root directory containing the AeBAD_S dataset.
        category: Category name.
        domain_shift: Domain shift category for test split
            (``"same"``, ``"view"``).
        augmentations: Augmentations applied to input images.
        split: Dataset split - ``Split.TRAIN`` or ``Split.TEST``.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "AeBAD_S",
        domain_shift: str = "same",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root = Path(root) if root is not None else Path("./datasets/AeBAD/AeBAD_S")
        self.category = category
        self.domain_shift = domain_shift
        self.split = split
        self.samples = make_aebad_s_dataset(
            root=self.root,
            split=self.split,
            domain_shift=self.domain_shift,
            extensions=IMG_EXTENSIONS,
        )


def make_aebad_s_dataset(
    root: str | Path,
    split: str | Split | None = None,
    domain_shift: str = "same",
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create AeBAD-S samples by parsing the directory structure.

    Args:
        root: Path to AeBAD_S dataset root directory.
        split: Dataset split (``Split.TRAIN`` or ``Split.TEST``).
        domain_shift: Domain shift category for test split filtering.
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

        for sub_dir in train_good_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for img_path in sub_dir.glob("*"):
                if img_path.suffix in extensions:
                    samples_list.append({
                        "image_path": str(img_path),
                        "mask_path": "",
                        "label_index": LabelName.NORMAL,
                        "split": Split.TRAIN,
                    })

    else:
        test_dir = root / "test"
        if not test_dir.is_dir():
            msg = f"Test directory not found: {test_dir}"
            raise FileNotFoundError(msg)

        gt_dir = root / "ground_truth"

        for defect_dir in test_dir.iterdir():
            if not defect_dir.is_dir():
                continue
            defect_name = defect_dir.name

            domain_dir = defect_dir / domain_shift
            if not domain_dir.is_dir():
                continue

            for img_path in domain_dir.glob("*"):
                if img_path.suffix not in extensions:
                    continue

                is_normal = defect_name == "good"
                mask_path = ""

                if not is_normal and gt_dir.is_dir():
                    mask_file = gt_dir / defect_name / domain_shift / img_path.name
                    if mask_file.is_file():
                        mask_path = str(mask_file)

                samples_list.append({
                    "image_path": str(img_path),
                    "mask_path": mask_path,
                    "label_index": LabelName.NORMAL if is_normal else LabelName.ABNORMAL,
                    "split": Split.TEST,
                })

    if not samples_list:
        msg = f"Found 0 images in {root} with split={split}, domain_shift={domain_shift}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Validate mask-image filename matching for anomalous samples
    abnormal_with_mask = samples.loc[
        (samples.label_index == LabelName.ABNORMAL) & (samples.mask_path != "")
    ]
    if len(abnormal_with_mask):
        if not abnormal_with_mask.apply(
            lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1,
        ).all():
            msg = (
                "Mismatch between anomalous images and ground truth masks. "
                "Make sure mask files follow the same naming convention "
                "as the anomalous images."
            )
            raise MisMatchError(msg)

    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    return samples
