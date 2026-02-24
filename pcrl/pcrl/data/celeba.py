"""CelebA dataset for PCRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.utils.config import PurposeSpec


class CelebADataset(PCRLDataset):
    """CelebA dataset for face attribute prediction.

    Large-scale face attributes dataset with 40 binary attributes.
    Commonly used for fairness research with gender as the sensitive attribute.

    Task labels are organized by purpose:
        - Hair color: Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair
        - Expression: Smiling, Mouth_Slightly_Open, Narrow_Eyes
        - Accessories: Eyeglasses, Wearing_Hat, Wearing_Earrings, Wearing_Necklace

    Sensitive attributes:
        - Male: Binary gender attribute
        - Young: Age-related attribute
        - Attractive: Can be sensitive for some applications

    Note: This implementation expects pre-extracted features or works with
    torchvision's CelebA dataset for image loading.
    """

    # Hair color attributes (for hair_color purpose)
    HAIR_COLOR_ATTRIBUTES = [
        "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald",
    ]

    # Expression attributes (for expression purpose)
    EXPRESSION_ATTRIBUTES = [
        "Smiling", "Mouth_Slightly_Open", "Narrow_Eyes", "Arched_Eyebrows",
    ]

    # Accessory attributes (for accessories purpose)
    ACCESSORY_ATTRIBUTES = [
        "Eyeglasses", "Wearing_Hat", "Wearing_Earrings", "Wearing_Necklace",
        "Wearing_Necktie",
    ]

    # Default task attributes (all usable as tasks)
    TASK_ATTRIBUTES = (
        HAIR_COLOR_ATTRIBUTES + EXPRESSION_ATTRIBUTES + ACCESSORY_ATTRIBUTES +
        ["Attractive", "High_Cheekbones", "Oval_Face", "Pointy_Nose", "Wearing_Lipstick"]
    )

    # Sensitive attributes
    SENSITIVE_ATTRIBUTES = ["Male", "Young", "Attractive"]

    # All 40 attributes in order
    ALL_ATTRIBUTES = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
        "Wearing_Necktie", "Young",
    ]

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
        feature_extractor: str | None = None,
        task_attrs: list[str] | None = None,
        sensitive_attrs: list[str] | None = None,
    ) -> None:
        """Initialize the CelebA dataset.

        Args:
            purposes: List of purpose specifications.
            root: Root directory for dataset storage.
            split: Dataset split ("train", "val", or "test").
            download: Whether to download the dataset if not present.
            transform: Optional transform to apply to images/features.
            feature_extractor: Name of pre-trained model for feature extraction.
                Options: "resnet18", "resnet50", "vit". If None, uses raw pixels.
            task_attrs: List of attributes to use as task labels.
                Defaults to TASK_ATTRIBUTES.
            sensitive_attrs: List of attributes to use as sensitive attributes.
                Defaults to SENSITIVE_ATTRIBUTES.
        """
        self.feature_extractor_name = feature_extractor
        self.task_attrs = task_attrs or self.TASK_ATTRIBUTES
        self.sensitive_attr_names = sensitive_attrs or self.SENSITIVE_ATTRIBUTES

        # Will be set in _load_data
        self._celeba_dataset: Dataset | None = None
        self._features: torch.Tensor | None = None
        self._feature_extractor: Any = None

        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        """Load and preprocess the CelebA dataset."""
        try:
            from torchvision.datasets import CelebA
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "torchvision is required for CelebA dataset. "
                "Install with: pip install torchvision"
            )

        # Map split names
        split_map = {"train": "train", "val": "valid", "test": "test"}
        celeba_split = split_map[self.split]

        # Image preprocessing
        image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load CelebA
        self._celeba_dataset = CelebA(
            root=self.root,
            split=celeba_split,
            target_type="attr",
            transform=image_transform,
            download=self.download,
        )

        # Set up feature extractor if specified
        if self.feature_extractor_name:
            self._setup_feature_extractor()

        # Get attribute indices
        self._task_attr_indices = [
            self.ALL_ATTRIBUTES.index(attr) for attr in self.task_attrs
        ]
        self._sensitive_attr_indices = [
            self.ALL_ATTRIBUTES.index(attr) for attr in self.sensitive_attr_names
        ]

        # Determine feature dimension
        if self.feature_extractor_name:
            feature_dim = self._get_feature_dim()
        else:
            feature_dim = 3 * 224 * 224  # Flattened image

        # Set dataset info
        self.info = DatasetInfo(
            name="celeba",
            num_features=feature_dim,
            task_labels={attr: 2 for attr in self.task_attrs},
            sensitive_attrs={attr: 2 for attr in self.sensitive_attr_names},
            num_samples=len(self._celeba_dataset),
        )

    def _setup_feature_extractor(self) -> None:
        """Set up the feature extraction model."""
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for feature extraction")

        if self.feature_extractor_name == "resnet18":
            model = models.resnet18(pretrained=True)
            self._feature_extractor = torch.nn.Sequential(
                *list(model.children())[:-1],
                torch.nn.Flatten(),
            )
        elif self.feature_extractor_name == "resnet50":
            model = models.resnet50(pretrained=True)
            self._feature_extractor = torch.nn.Sequential(
                *list(model.children())[:-1],
                torch.nn.Flatten(),
            )
        else:
            raise ValueError(f"Unknown feature extractor: {self.feature_extractor_name}")

        self._feature_extractor.eval()
        for param in self._feature_extractor.parameters():
            param.requires_grad = False

    def _get_feature_dim(self) -> int:
        """Get the output dimension of the feature extractor."""
        if self.feature_extractor_name in ("resnet18",):
            return 512
        elif self.feature_extractor_name in ("resnet50",):
            return 2048
        else:
            return 3 * 224 * 224

    def __len__(self) -> int:
        """Return the number of samples."""
        if self._celeba_dataset is None:
            return 0
        return len(self._celeba_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample."""
        if self._celeba_dataset is None:
            raise RuntimeError("Dataset not loaded")

        image, attrs = self._celeba_dataset[idx]

        # Extract features
        if self._feature_extractor is not None:
            with torch.no_grad():
                features = self._feature_extractor(image.unsqueeze(0)).squeeze(0)
        else:
            features = image.flatten()

        if self.transform is not None:
            features = self.transform(features)

        # Convert attributes from {-1, 1} to {0, 1}
        attrs = (attrs + 1) // 2

        # Extract task labels
        task_labels = {
            attr: attrs[self._task_attr_indices[i]]
            for i, attr in enumerate(self.task_attrs)
        }

        # Extract sensitive attributes
        sensitive_attrs = {
            attr: attrs[self._sensitive_attr_indices[i]]
            for i, attr in enumerate(self.sensitive_attr_names)
        }

        return {
            "features": features,
            "task_labels": task_labels,
            "sensitive_attrs": sensitive_attrs,
        }


class CelebAFeaturesDataset(PCRLDataset):
    """CelebA dataset with pre-extracted features.

    Use this when you have pre-computed features stored as tensors,
    which is much faster than extracting features on-the-fly.
    """

    def __init__(
        self,
        purposes: list[PurposeSpec],
        features_path: str,
        labels_path: str,
        root: str = "data",
        split: str = "train",
        download: bool = False,
        transform: Any | None = None,
        task_attrs: list[str] | None = None,
        sensitive_attrs: list[str] | None = None,
    ) -> None:
        """Initialize with pre-extracted features.

        Args:
            purposes: List of purpose specifications.
            features_path: Path to pre-extracted features tensor.
            labels_path: Path to labels tensor.
            root: Root directory (not used, kept for API compatibility).
            split: Dataset split embedded in the feature files.
            download: Not used for pre-extracted features.
            transform: Optional transform to apply to features.
            task_attrs: List of attributes to use as task labels.
            sensitive_attrs: List of attributes to use as sensitive attributes.
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.task_attrs = task_attrs or CelebADataset.TASK_ATTRIBUTES
        self.sensitive_attr_names = sensitive_attrs or CelebADataset.SENSITIVE_ATTRIBUTES

        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        """Load pre-extracted features and labels."""
        features_path = Path(self.features_path)
        labels_path = Path(self.labels_path)

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.features = torch.load(features_path)
        all_labels = torch.load(labels_path)

        # Get attribute indices
        task_indices = [
            CelebADataset.ALL_ATTRIBUTES.index(attr) for attr in self.task_attrs
        ]
        sensitive_indices = [
            CelebADataset.ALL_ATTRIBUTES.index(attr)
            for attr in self.sensitive_attr_names
        ]

        # Extract task and sensitive labels
        self.task_labels = {
            attr: all_labels[:, task_indices[i]].long()
            for i, attr in enumerate(self.task_attrs)
        }
        self.sensitive_attrs = {
            attr: all_labels[:, sensitive_indices[i]].long()
            for i, attr in enumerate(self.sensitive_attr_names)
        }

        self.info = DatasetInfo(
            name="celeba_features",
            num_features=self.features.shape[1],
            task_labels={attr: 2 for attr in self.task_attrs},
            sensitive_attrs={attr: 2 for attr in self.sensitive_attr_names},
            num_samples=len(self.features),
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample."""
        features = self.features[idx]

        if self.transform is not None:
            features = self.transform(features)

        return {
            "features": features,
            "task_labels": {k: v[idx] for k, v in self.task_labels.items()},
            "sensitive_attrs": {k: v[idx] for k, v in self.sensitive_attrs.items()},
        }


def get_celeba_purposes() -> list[PurposeSpec]:
    """Get standard purpose specifications for the CelebA dataset.

    Returns three purposes demonstrating different fairness constraints:
    1. hair_color: Predict hair color attributes while hiding gender and age
    2. expression: Predict expression attributes while hiding gender and attractiveness
    3. accessories: Predict accessory attributes while hiding gender, age, and attractiveness
    """
    return [
        PurposeSpec(
            name="hair_color",
            allowed_tasks=["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald"],
            disallowed_attrs=["Male", "Young"],
            task_type="classification",
        ),
        PurposeSpec(
            name="expression",
            allowed_tasks=["Smiling", "Mouth_Slightly_Open", "Narrow_Eyes", "Arched_Eyebrows"],
            disallowed_attrs=["Male", "Attractive"],
            task_type="classification",
        ),
        PurposeSpec(
            name="accessories",
            allowed_tasks=["Eyeglasses", "Wearing_Hat", "Wearing_Earrings", "Wearing_Necklace"],
            disallowed_attrs=["Male", "Young", "Attractive"],
            task_type="classification",
        ),
    ]
