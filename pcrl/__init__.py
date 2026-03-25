"""PCRL: Purpose-Conditioned Representation Learning.

A framework for learning fair representations that are conditioned on
data processing purposes, enabling purpose-specific information filtering.
"""

__version__ = "0.1.0"

from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.utils.config import (
    ExperimentConfig,
    PCRLConfig,
    TrainingConfig,
    load_config,
    load_config_from_yaml,
    save_config_to_yaml,
)
from pcrl.models.conditioning import (
    AttentionConditioner,
    ConcatConditioner,
    FiLMConditioner,
    build_conditioner,
)
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import MultiTaskHead, PurposeTaskHeads, TaskHead
from pcrl.models.auditor import (
    Auditor,
    AuditorPool,
    MultiAttributeAuditor,
    PostHocAuditorSuite,
    PurposeAuditors,
)

__all__ = [
    # Version
    "__version__",
    # Purposes
    "PurposeSpec",
    "PurposeRegistry",
    # Config
    "PCRLConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_config",
    "load_config_from_yaml",
    "save_config_to_yaml",
    # Conditioning
    "FiLMConditioner",
    "ConcatConditioner",
    "AttentionConditioner",
    "build_conditioner",
    # Models
    "PurposeConditionedEncoder",
    "TaskHead",
    "MultiTaskHead",
    "PurposeTaskHeads",
    "Auditor",
    "AuditorPool",
    "MultiAttributeAuditor",
    "PurposeAuditors",
    "PostHocAuditorSuite",
]
