"""Model modules for PCRL."""

from pcrl.models.auditor import (
    Auditor,
    AuditorConfig,
    AuditorPool,
    GradientReversalFunction,
    MultiAttributeAuditor,
    PostHocAuditorSuite,
    PurposeAuditors,
    gradient_reversal,
)
from pcrl.models.conditioning import (
    AttentionConditioner,
    ConcatConditioner,
    FiLMConditioner,
    build_conditioner,
)
from pcrl.models.encoder import (
    PurposeConditionedEncoder,
    StandardEncoder,
)
from pcrl.models.task_head import (
    MultiTaskHead,
    PurposeTaskHeads,
    TaskHead,
)

__all__ = [
    # Encoder
    "PurposeConditionedEncoder",
    "StandardEncoder",
    # Conditioning
    "FiLMConditioner",
    "ConcatConditioner",
    "AttentionConditioner",
    "build_conditioner",
    # Task heads
    "TaskHead",
    "MultiTaskHead",
    "PurposeTaskHeads",
    # Auditors
    "Auditor",
    "AuditorConfig",
    "AuditorPool",
    "MultiAttributeAuditor",
    "PurposeAuditors",
    "PostHocAuditorSuite",
    "GradientReversalFunction",
    "gradient_reversal",
]
