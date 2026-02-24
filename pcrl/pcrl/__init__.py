"""PCRL: Purpose-Conditioned Representation Learning.

A framework for learning fair representations that are conditioned on
data processing purposes, enabling purpose-specific information filtering.
"""

__version__ = "0.1.0"

from pcrl.data import (
    AdultDataset,
    CelebADataset,
    DatasetInfo,
    PCRLDataset,
    collate_pcrl_batch,
    get_adult_purposes,
    get_celeba_purposes,
)
from pcrl.evaluation import (
    CVRMetric,
    CVRResult,
    PostHocProber,
    ProbeResult,
    evaluate_all_purposes,
)
from pcrl.models import (
    Auditor,
    MultiAttributeAuditor,
    MultiTaskHead,
    PurposeAuditors,
    PurposeConditionedEncoder,
    PurposeTaskHeads,
    TaskHead,
)
from pcrl.training import (
    PCRLLoss,
    PCRLTrainer,
    TrainingMetrics,
    create_trainer,
)
from pcrl.utils import (
    ExperimentConfig,
    PCRLConfig,
    PurposeSpec,
    TrainingConfig,
    load_config_from_yaml,
    save_config_to_yaml,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "PurposeSpec",
    "PCRLConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_config_from_yaml",
    "save_config_to_yaml",
    # Data
    "PCRLDataset",
    "DatasetInfo",
    "collate_pcrl_batch",
    "AdultDataset",
    "get_adult_purposes",
    "CelebADataset",
    "get_celeba_purposes",
    # Models
    "PurposeConditionedEncoder",
    "TaskHead",
    "MultiTaskHead",
    "PurposeTaskHeads",
    "Auditor",
    "MultiAttributeAuditor",
    "PurposeAuditors",
    # Training
    "PCRLLoss",
    "PCRLTrainer",
    "TrainingMetrics",
    "create_trainer",
    # Evaluation
    "CVRMetric",
    "CVRResult",
    "evaluate_all_purposes",
    "PostHocProber",
    "ProbeResult",
]
